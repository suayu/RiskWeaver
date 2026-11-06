"""真实百炼单次规划探针；只输出非敏感计划、耗时与验证状态。"""
import json
from pathlib import Path
import sys
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.profile_planner import rank_profile_candidates
from test_ego_profile import environment,pipeline

path=Path(sys.argv[1])
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists():
    raise RuntimeError('probe output already exists')
planner=LLMAdversarialPlanner(model_name='qwen3.5-plus',model_names=['qwen3.5-plus'],provider='dashscope',attack_mode='trajectory_only')
if not planner.available:
    raise RuntimeError('DASHSCOPE_API_KEY unavailable')
planner.client=planner.client.with_options(timeout=60.,max_retries=0)
env=environment()
context=pipeline().context(env,.5)
start=time.monotonic()
plan=rank_profile_candidates(planner,env.get_state_for_planning(),['Choose a feasible candidate to probe ego braking response.'],context)
record=dict(provider=planner.provider,model=planner.model_name,input_kind='synthetic_validated_interaction_fixture',
            elapsed_seconds=time.monotonic()-start,plan=plan,trace=planner.last_trace,
            passed=bool(plan and plan.get('attack') and not planner.last_request_failed))
path.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
raise SystemExit(0 if record['passed'] else 1)
