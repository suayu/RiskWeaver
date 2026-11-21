"""冻结新版单组20场协议。默认仅预检，冒烟终态通过后才允许启动。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path("/home2/zhaoyx/scenario-dreamer")
HERE = ROOT / "work/idm_nonnegative_20261113"
NAME = "skills_idm_fixed20_20261113_r2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    smoke = ROOT / "experiments/idm_nonnegative_fixture060_20261113_r1/skills_joint_dual"
    exit_status = json.loads((smoke / "exit_status.json").read_text())
    records = json.loads((smoke / "results.json").read_text())["episodes"]
    if exit_status.get("exit_code") != 0 or len(records) != 1:
        raise ValueError("idm_fixture_smoke_incomplete")
    if any(r.get("runtime_failure") or r.get("generation_failure") or not r.get("executed_steps") for r in records):
        raise ValueError("idm_fixture_smoke_failed")
    # 核验真实停车积分诊断，排除负速度翻转和新增API；不把重放当新Skills效果。
    import math
    states = [json.loads(line) for line in (smoke / "movies/scenario_060/execution_trace.jsonl").read_text().splitlines()]
    states = [row for row in states if row["kind"] == "state"]
    if len(states) != records[0]["executed_steps"] + 1:
        raise ValueError("fixture_trace_incomplete")
    if max(math.hypot(*row["ego"][2:4]) for row in states) > 17:
        raise ValueError("fixture_ego_speed_abnormal")
    if any(row["ego"][2] * math.cos(row["ego"][4]) + row["ego"][3] * math.sin(row["ego"][4]) < -1e-5 for row in states):
        raise ValueError("fixture_negative_forward_speed")
    if any(p.stat().st_size for p in smoke.parent.rglob("*api_calls.jsonl")):
        raise ValueError("fixture_unexpected_new_api")
    old = ROOT / "experiments/skills_gates_observe20_20261113_r4"
    frozen = HERE / NAME
    experiment = ROOT / "experiments" / NAME
    # 旧目录只用于初次准备；启动时完全依赖本轮已冻结协议和清单。
    if args.prepare:
        command = json.loads((old / "protocol.json").read_text())["commands"]["skills_joint_dual"]
        for flag, value in (("--manifest", frozen / "manifest.json"), ("--output-dir", experiment / "skills_joint_dual")):
            command[command.index(flag) + 1] = str(value)
        command[command.index("--reachability-preflight") + 1] = "on"
        names = list(json.loads((old / "source_sha256.json").read_text()))
        names += ["policies/attack_recovery.py", "policies/static_attack_catalog.py",
                  "policies/hybrid_skills_planner.py", "work/gates_observe_20261113/compare_campaign.py", "work/gates_observe_20261113/prepare_acceptance.py", "work/idm_nonnegative_20261113/prepare_acceptance.py", "work/idm_nonnegative_20261113/test_fix.py"]
        names += ["work/idm_nonnegative_20261113/prepare_acceptance_r2.py"]
        names += [f"skills/riskweaver/{name}/SKILL.md" for name in ("rw-current-fact-reasoning", "rw-case-evidence", "rw-source-facts", "rw-static-fallback")]
    else:
        command = json.loads((frozen / "protocol.json").read_text())["commands"]["skills_joint_dual"]
        names = list(json.loads((frozen / "source_sha256.json").read_text()))
    hashes = {name: digest(ROOT / name) for name in sorted(set(names))}
    if args.prepare:
        frozen.mkdir(exist_ok=False)
        shutil.copyfile(old / "manifest.json", frozen / "manifest.json")
        for name in hashes:
            target = frozen / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        (frozen / "source_sha256.json").write_text(json.dumps(hashes, indent=2))
        protocol = dict(commands={"skills_joint_dual": command}, planned_attempts=20,
                        original_manifest_sha256=digest(old / "manifest.json"), source_count=len(hashes),
                        speed_cap=80., acceleration_cap=8., original_backend_retained=True,
                        policy="joint_and_area_gates_observe_only_dynamic_first",
                        idm_nonnegative_speed=True, physical_quality_control_required=True,
                        fixture_scope="initial_plan_only_no_new_api_not_new_skills_coverage",
                        joint_gate_enforced=False, area_gate_enforced=False, zero_area_scenes_included=True,
                        required_area_comparison="route_grid_and_commonroad_circumscribed_circle_every_saved_native_source",
                        collision_rate_basis="ego_only_background_events_recorded",
                        required_effective_scenes=18, required_ego_collision_scenes=11,
                        max_physical_requests_per_scene=4, model="deepseek-flash",
                        prerequisite_fixture_smoke_sha256=digest(smoke / "results.json"))
        (frozen / "protocol.json").write_text(json.dumps(protocol, indent=2))
        with (frozen / "prepare_only.log").open("x") as log:
            subprocess.run(command + ["--prepare-only"], cwd=ROOT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=".:safe-sim:safe-sim/trajdata/src"),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
    if json.loads((frozen / "source_sha256.json").read_text()) != hashes:
        raise ValueError("source_changed_since_preparation")
    if digest(frozen / "manifest.json") != json.loads((frozen / "protocol.json").read_text())["original_manifest_sha256"]:
        raise ValueError("frozen_twenty_scene_inventory_changed")
    if not args.launch:
        print(json.dumps(dict(preflight=True, launched=False, source_count=len(hashes))))
        return
    if shutil.disk_usage(ROOT).free < 5.5 * 2**30:
        raise RuntimeError("insufficient_disk")
    experiment.mkdir(exist_ok=False)
    for name in ("manifest.json", "source_sha256.json", "protocol.json"):
        shutil.copyfile(frozen / name, experiment / name)
    shutil.copytree(frozen / "source", experiment / "source_snapshot")
    with (experiment / "run.log").open("x") as log:
        worker = subprocess.Popen(["bash", "scripts/run_with_llm_provider.sh", "deepseek", *command],
                                  cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=".:safe-sim:safe-sim/trajdata/src",
                                                     OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
                                                     LLM_MODEL_NAME="deepseek-flash", LLM_MODEL_NAMES="deepseek-flash"),
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (experiment / "worker.pid").write_text(str(worker.pid))
    print(json.dumps(dict(pid=worker.pid, output=str(experiment), started=time.time())))


if __name__ == "__main__":
    main()
