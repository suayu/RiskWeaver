"""使用失败 R2 的同一初始场景验证百炼完整攻击规划 JSON 契约。"""

import json
import os
import pickle
import time
from pathlib import Path

from policies.llm_adversarial_planner import LLMAdversarialPlanner


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
RUN = ROOT / "experiments/riskweaver_profile_formal_idm_rl_20260924_r2/idm_profile_off"
trace_path = RUN / "movies/scenario_000/execution_trace.jsonl"
state = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
manifest = json.loads((ROOT / "work/riskweaver_profile_formal_manifest_20260923.json").read_text())
with Path(manifest["scenario_files"][0]).open("rb") as handle:
    scene = pickle.load(handle)

agents = []
for agent_id, (agent_state, active) in enumerate(zip(state["agents"], state["active"])):
    if not active:
        continue
    agents.append({
        "id": agent_id,
        "type": 0,
        "state": [float(agent_state[i]) for i in (0, 1, 2, 3, 4)],
        "history": [],
    })
env_state = {
    "ego_state": [float(state["ego"][i]) for i in (0, 1, 2, 3, 4)],
    "route": [[float(point[0]), float(point[1])] for point in scene["route"]],
    "agents": agents,
    "static_obstacles": [],
    "current_step": 0,
    "history_order": "oldest_to_newest",
}

model_name = os.environ.get("RISKWEAVER_SMOKE_MODEL", "qwen-plus-0112")
planner = LLMAdversarialPlanner(
    provider="dashscope",
    model_name=model_name,
    model_names=[model_name],
    attack_mode="trajectory_only",
)
started = time.perf_counter()
plan = planner.generate_attack_plan(
    env_state,
    [
        "Generate a physically executable adversarial interaction that probes the ego driving strategy. "
        "The attacker may leave the road topology. Preserve non-target background safety and a positive "
        "theoretical drivable region for the ego."
    ],
)
elapsed = time.perf_counter() - started
payload = {
    "model": model_name,
    "elapsed_seconds": elapsed,
    "plan_returned": plan is not None,
    "attack": bool(plan and plan.get("attack")),
    "plan": plan,
    "last_error": planner.last_error,
    "last_request_failed": planner.last_request_failed,
    "trace": planner.last_trace,
}
safe_model_name = model_name.replace("/", "_")
output = ROOT / f"work/full_planner_smoke_20260924_{safe_model_name}.json"
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: payload[key] for key in (
    "model", "elapsed_seconds", "plan_returned", "attack", "last_error", "last_request_failed"
)}, ensure_ascii=False, indent=2))
raise SystemExit(0 if payload["plan_returned"] and payload["attack"] else 2)
