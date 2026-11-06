"""固定单模型的 JSON 契约探针；不记录提示、响应正文或凭据。"""
import json
import time
from pathlib import Path

from policies.llm_adversarial_planner import LLMAdversarialPlanner


MODEL = "qwen3-8b"
OUTPUT = Path("work/qwen3_8b_schema_probe_20260925.json")
started = time.monotonic()
planner = LLMAdversarialPlanner(
    model_name=MODEL, model_names=[MODEL], provider="dashscope", attack_mode="trajectory_only"
)
payload = {"request_id":"schema-probe-20260925","task":"Return exactly one JSON object with attack false, attack_target_id -1, empty anchors, strategy none, duration 0, and a short no_attack_reason."}
result = {"model": MODEL, "started": time.time(), "available": bool(planner.available)}
try:
    _, response = planner._request_attack_plan(json.dumps(payload, ensure_ascii=False))
    result.update(ok=True, response_keys=sorted(map(str, response.keys())),
                  attempted_models=list(planner.last_attempted_model_names),
                  successful_model=planner.last_successful_model_name)
except Exception as error:
    result.update(ok=False, error_type=type(error).__name__,
                  attempted_models=list(getattr(planner,"last_attempted_model_names",[])),
                  failures=list(getattr(planner,"last_model_failures",[])))
finally:
    result["elapsed_seconds"] = time.monotonic()-started
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
raise SystemExit(0 if result.get("ok") else 2)
