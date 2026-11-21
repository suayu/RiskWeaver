"""将 3 秒动态计划接入完整 Skills 工作器，保留原入口供历史复现。"""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

from anchor_execution import ANCHOR_TIMES, interpolate_joint, window_steps
from avoidance_witness import add_evidence, apply_verdict


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


@contextmanager
def install(mode, runner, planner, generator, simulator, recipes, projection, rejection,
            witness_accel=None):
    """在同一进程装配时域及后端，退出时恢复所有钩子。"""
    with ExitStack() as stack:
        def patch(obj, name, value):
            stack.callback(setattr, obj, name, getattr(obj, name))
            setattr(obj, name, value)

        original_context = planner.build_context
        original_recipe = recipes.proposals_for_actor
        original_plan = generator.plan_and_inject
        original_step = generator.step
        original_prepare = simulator.prepare_background_traffic
        original_overrides = runner.group_overrides
        original_project = projection.project_joint

        @wraps(original_context)
        def build_context(self, env, *args, **kwargs):
            context = original_context(self, env, *args, **kwargs)
            duration = window_steps(float(env.dt))
            # 在派生证据、目录哈希和模型请求形成前替换候选时域。
            context["candidates"] = [dict(candidate, duration=duration)
                                     for candidate in context["candidates"]]
            return context

        @wraps(original_recipe)
        def full_recipe(ego, actor, target_id, dt=.1, duration=10, **kwargs):
            return original_recipe(ego, actor, target_id, dt, window_steps(float(dt)), **kwargs)

        @wraps(original_plan)
        def plan(self, env):
            prior = env.attack_intent
            outcome = original_plan(self, env)
            intent = env.attack_intent
            if intent is not None and intent is not prior:
                duration = window_steps(float(env.dt))
                if len(intent["anchors"]) != 4 or self.attack_duration != duration or intent["duration"] != duration:
                    raise ValueError("accepted_plan_does_not_cover_full_anchor_window")
                intent["anchor_times_s"] = list(ANCHOR_TIMES)
                trace = getattr(self, "query_trace", None)
                if trace is not None:
                    trace.append(dict(kind="anchor_execution_window", step=int(env.current_step),
                                      request_id=intent.get("request_id"), target_id=intent["target_id"],
                                      duration_steps=duration, anchor_times_s=list(ANCHOR_TIMES), backend=mode))
            return outcome

        @wraps(original_step)
        def step(self, env, current_t):
            intent = env.attack_intent
            if intent is not None and current_t < self.last_attack_frame + self.attack_duration:
                if env.agent_active[intent["target_id"]]:
                    # 有效目标的整个 3 秒计划不被定时或视野边界重规划打断。
                    return True
                self.attack_duration = current_t - self.last_attack_frame
            return original_step(self, env, current_t)

        @wraps(original_prepare)
        def prepare(self):
            duration = window_steps(float(self.dt))
            if int(self.cfg.sim.traffic_model.prediction_horizon) < duration:
                raise ValueError("prediction_horizon_shorter_than_anchor_window")
            if mode == "interpolation":
                if self.traffic_backend != "external_joint" or self.diffusion_controller is not None:
                    raise ValueError("interpolation_requires_no_diffusion_controller")
                return interpolate_joint(self)
            joint = original_prepare(self)
            intent = self.attack_intent
            if intent is not None:
                remaining = duration - (self.current_step - intent["source_step"])
                reused = int(joint.metadata.get("replan_cache_offset", 0))
                if (not joint.metadata.get("anchor_guidance_active")
                        or joint.metadata.get("anchor_guidance_target_id") != intent["target_id"]
                        or joint.metadata.get("anchor_guidance_valid_steps", 0) - reused < remaining):
                    raise rejection("diffusion_did_not_cover_remaining_anchor_window")
            return joint

        patch(planner, "build_context", build_context)
        patch(recipes, "proposals_for_actor", full_recipe)
        patch(generator, "plan_and_inject", plan)
        patch(generator, "step", step)
        patch(simulator, "prepare_background_traffic", prepare)
        if witness_accel is not None:
            original_evaluate = generator.evaluate_reachability
            original_assess = runner.assess_reachability_execution

            def evaluate(self, env):
                event = original_evaluate(self, env)
                cfg = env.cfg.sim.evaluation.reachability
                if cfg.execution_gate_policy == "positive_area":
                    return add_evidence(event, env, witness_accel)
                return event

            def assess(event, policy):
                return apply_verdict(original_assess(event, policy), event, policy)

            patch(generator, "evaluate_reachability", evaluate)
            patch(runner, "assess_reachability_execution", assess)
        if mode == "interpolation":
            def overrides(*args, **kwargs):
                return original_overrides(*args, **kwargs) + [
                    "sim.traffic_model.backend=external_joint", "sim.traffic_model.device=cpu"]

            def project(joint, *args, **kwargs):
                if joint is not None and joint.metadata.get("backend") == "kinematic_interpolation":
                    # 目标轨迹已做差分校验与必要的投影，避免二次投影漂移。
                    # 后续统一联合硬门控、执行首帧复核和正面积门控照常执行。
                    audit = dict(method="checked_kinematic_interpolation", vehicles=[], elapsed_seconds=0.)
                    return replace(joint, metadata=dict(joint.metadata, dynamics_projection=audit))
                return original_project(joint, *args, **kwargs)

            patch(runner, "group_overrides", overrides)
            patch(projection, "project_joint", project)
        yield


@contextmanager
def cpu_interpolation_resource():
    """CPU 插值专用装配，保留完整规划、预算、日志与评分工作器。"""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("cpu_interpolation_requires_hidden_cuda")
    sys.path[:0] = [str(ROOT / "work" / name) for name in
                   ("skills_shared_resource_20261028", "skills_provider_compat_20261028")]
    import run_capped

    original = run_capped.main

    def run():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--resource-cap-gib", type=float, required=True)
        parser.add_argument("--resource-reserve-gib", type=float, default=3.)
        parser.add_argument("--resource-fixture", choices=("off",), default="off")
        parser.add_argument("--resource-observe-sampling", choices=("off",), default="off")
        _, rest = parser.parse_known_args()
        import run_controlled

        previous = sys.argv
        try:
            sys.argv = [previous[0], *rest]
            return run_controlled.main()
        finally:
            sys.argv = previous

    run_capped.main = run
    try:
        yield
    finally:
        run_capped.main = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-backend", choices=("diffusion", "interpolation"), default="diffusion")
    parser.add_argument("--interpolation-cpu", action="store_true")
    parser.add_argument("--witness-max-accel-mps2", type=float)
    args, rest = parser.parse_known_args()
    required = {"--skills": "on", "--skill-level": "full", "--incremental-skills": "on",
                "--group": "riskweaver_profile_off", "--policy": "idm"}
    if any(rest.count(flag) != 1 or rest[rest.index(flag)+1:rest.index(flag)+2] != [value]
           for flag, value in required.items()):
        parser.error("the anchor worker requires the full Skills IDM configuration")
    if args.interpolation_cpu and args.anchor_backend != "interpolation":
        parser.error("--interpolation-cpu requires interpolation")
    if args.witness_max_accel_mps2 is not None and not 2. < args.witness_max_accel_mps2 <= 6.:
        parser.error("witness acceleration must be an explicit physical cap in (2, 6] m/s^2")
    out = Path(rest[rest.index("--output-dir") + 1])
    prepare_only = "--prepare-only" in rest
    if not prepare_only:
        hashes = json.loads((out.parent / "source_sha256.json").read_text())
        for path in (Path(__file__).resolve(), HERE / "anchor_execution.py", HERE / "avoidance_witness.py"):
            if hashes.get(path.relative_to(ROOT).as_posix()) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("anchor_execution_sources_not_frozen")

    sys.path[:0] = [str(ROOT), str(ROOT / "work/skills_incremental_evidence_20261025")]
    from experiments.comparison import run_e1_group as runner
    from policies import trajectory_projection as projection
    from policies.joint_safety import NoSafeJointCandidate
    from policies.skills_planner import SkillsAttackPlanner
    import active_window_recipes

    delegate = ROOT / "work/skills_acceptance_20261109/run_idm_heading.py"
    old_argv = sys.argv
    try:
        sys.argv = [str(delegate), *rest]
        with ExitStack() as stack:
            if args.interpolation_cpu:
                stack.enter_context(cpu_interpolation_resource())
            stack.enter_context(install(args.anchor_backend, runner, SkillsAttackPlanner,
                                        runner.AdversarialScenarioGenerator, runner.Simulator,
                                        active_window_recipes, projection, NoSafeJointCandidate,
                                        witness_accel=args.witness_max_accel_mps2))
            runpy.run_path(str(delegate), run_name="__main__")
    finally:
        sys.argv = old_argv
        if not prepare_only and out.is_dir():
            with (out / "anchor_execution_protocol.json").open("x", encoding="utf-8") as file:
                json.dump(dict(backend=args.anchor_backend, cpu=args.interpolation_cpu,
                               anchor_times_s=list(ANCHOR_TIMES), full_anchor_window=True,
                               background_prediction="constant_velocity" if args.anchor_backend == "interpolation" else "diffusion",
                               avoidance_witness_max_accel_mps2=args.witness_max_accel_mps2,
                               gate_policy_unchanged=args.witness_max_accel_mps2 is None), file, indent=2)


if __name__ == "__main__":
    main()
