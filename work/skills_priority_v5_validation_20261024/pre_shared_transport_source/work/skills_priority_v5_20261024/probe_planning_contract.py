"""一个冻结开发状态的真实百炼契约检查；没有扩散、实际控制或闭环成功声明。"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from model_pool_protocol import inspect_model_source
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from priority_planner import PrioritySkillsPlanner


def main():
    if os.getenv('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('planning_probe_requires_GPU_hidden')
    out = ROOT/'experiments/skills_priority_v5_planning_contract_scene182_20261024_r1'
    if out.exists():
        raise FileExistsError('preserve_existing_planning_probe')
    source = ROOT/'experiments/e1_five_arm_idm_10_modelpool_20261014_r1/formal/riskweaver_profile_off'
    trace = next(p for p in source.rglob('execution_trace.jsonl') if p.parent.name == 'scenario_182')
    rows = [json.loads(x) for x in trace.read_text().splitlines()]
    request = next(r for r in rows if r['kind'] == 'llm_input')
    frame = next(r for r in rows if r['kind'] == 'state' and r['step'] == request['step'])
    state = request['state']
    cfg = OmegaConf.load(ROOT/'experiments/skills_modelpool_recovery_development10_paired_20261023_r3/skills_full_shared_recovery/resolved_config.yaml')
    cfg.sim.llm.skills.root = str(HERE/'skills')
    model_source = inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    approved = json.loads((ROOT/'work/skills_recovery_paired_protocol_20261023_v4/protocol.json').read_text())['frozen_model_pool']
    active = [m for m in approved if m not in model_source['disabled_models']]
    pool = (['kimi-k3'] if 'kimi-k3' in active else []) + [m for m in active if m != 'kimi-k3']
    if not pool or not (ROOT/'work/bailian_pool_protection_confirmed_20261023.json').exists():
        raise ValueError('no_confirmed_free_active_pool')
    os.environ['LLM_MODEL_NAME'] = pool[0]
    os.environ['LLM_MODEL_NAMES'] = ','.join(pool)
    # 保存真实路线代理及其局限，不能把这个状态重放包装成真实地图/闭环验证。
    route = np.asarray(state['route'],float)
    env = NS(ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
        agent_active=np.asarray(frame['active']),current_step=request['step'],current_scene_id='scenario_182',
        scenario_dict={'route':route,'lanes':[route]},get_static_obstacles=lambda:[],cfg=cfg,dt=float(cfg.sim.dt),
        pending_joint_trajectory=None)
    env.get_state_for_planning = lambda:state
    planner = LLMAdversarialPlanner(model_name=pool[0],model_names=pool,provider='dashscope',attack_mode='trajectory_only')
    if not planner.available:
        raise ValueError('bailian_unavailable')
    planner.client = planner.client.with_options(max_retries=0)
    out.mkdir(exist_ok=False)
    files = [p for p in HERE.rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json'} and '__pycache__' not in p.parts]
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    launch = dict(provider='dashscope',preferred_model=pool[0],active_frozen_pool=pool,source_trace=str(trace),
        source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),source_sha256=hashes,
        free_protection_operator_confirmed=True,physical_request_limit=4,actual_execution_frames=0,
        scope='one frozen state; route-as-lanes proxy; no diffusion/control/performance/coverage claim',started_at=time.time())
    (out/'launch.json').write_text(json.dumps(launch,ensure_ascii=False,indent=2))
    calls=[]; create=planner.client.chat.completions.create
    def counted(*args,**kwargs):
        if len(calls)>=4:
            raise RuntimeError('planning_probe_physical_budget_exhausted')
        event=dict(request_index=len(calls)+1,model_requested=kwargs.get('model'),messages=kwargs.get('messages'),
            actual_extra_body=kwargs.get('extra_body'),started_at=time.time());calls.append(event);begin=time.monotonic()
        try:
            result=create(*args,**kwargs)
            event.update(model_returned=result.model,content=result.choices[0].message.content,
                usage=result.usage.model_dump() if result.usage else None)
            return result
        except Exception as error:
            event.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None));raise
        finally:
            event['elapsed_seconds']=time.monotonic()-begin
            with (out/'llm_calls.jsonl').open('a') as file:file.write(json.dumps(event,ensure_ascii=False)+'\n')
    planner.client.chat.completions.create=counted
    start=time.monotonic(); engine=PrioritySkillsPlanner(cfg.sim,4)
    plan=engine.plan(planner,env,state,request.get('instruction') or ['用背景车辆的离散攻击意图挑战自车，禁止静态障碍物。'])
    unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
    accepted=bool(plan and plan.get('attack'))
    result=dict(accepted_discrete_plan=accepted,plan=plan,planner_trace=planner.last_trace,
        planner_service_failure=planner.last_request_failed,physical_requests=len(calls),
        successful_responses=sum(bool(c.get('model_returned')) for c in calls),
        elapsed_seconds=time.monotonic()-start,source_unchanged=unchanged,
        actual_execution_frames=0,actual_attack_frames=0,D=None,
        scope=launch['scope'],goal_complete=False)
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    passed=accepted and not planner.last_request_failed and unchanged and len(calls)<=4
    (out/'exit_status.json').write_text(json.dumps(dict(exit_code=0 if passed else 1,planning_contract_only=True,finished_at=time.time())))
    print(json.dumps({k:v for k,v in result.items() if k not in ['plan','planner_trace']},ensure_ascii=False))
    return 0 if passed else 1


if __name__ == '__main__':sys.exit(main())
