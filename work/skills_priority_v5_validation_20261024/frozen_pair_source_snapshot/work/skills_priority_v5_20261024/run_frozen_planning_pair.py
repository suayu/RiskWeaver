"""同一原开发10状态比较v4和优先级v5；固定物理预算、公共传输适配、零GPU。"""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import hashlib
import json
import os
import sys
import time
import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT),str(HERE),str(ROOT/'work/skills_modelpool_budget_dev_20261022'),
               str(ROOT/'work/skills_recovery_v4_20261023'),str(ROOT/'work/skills_v2_compact_dev_20261020')]
from model_pool_protocol import inspect_model_source, verify_pool_transition
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from route_tail_preflight import RouteTailReadyPlanner
from priority_planner import PrioritySkillsPlanner
from bailian_transport import install_provider_compatibility


def save(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8');temp.replace(path)


def main():
    if os.getenv('CUDA_VISIBLE_DEVICES') != '':raise ValueError('CPU_contract_only_requires_GPU_hidden')
    out=ROOT/'experiments/skills_priority_v5_frozen_development10_pair_20261024_r1'
    if out.exists():raise FileExistsError('preserve_prior_planning_pair')
    old=ROOT/'work/skills_recovery_paired_protocol_20261023_v4'
    manifest=json.loads((old/'development10.json').read_text());ids=manifest['original_scenario_indices']
    assert len(ids)==10
    approved=json.loads((old/'protocol.json').read_text())['frozen_model_pool']
    if not (ROOT/'work/bailian_pool_protection_confirmed_20261023.json').exists():raise ValueError('free_protection_receipt_required')
    before_model=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    frozen=json.loads((old/'source_sha256.json').read_text())
    files=[ROOT/n for n in frozen if n!='policies/llm_adversarial_planner.py']
    files += [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.json'}]
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    sources=ROOT/'experiments/e1_five_arm_idm_10_modelpool_20261014_r1/formal/riskweaver_profile_off'
    source_by_id={int(p.parent.name.rsplit('_',1)[1]):p for p in sources.rglob('execution_trace.jsonl')}
    inputs=[]
    for idx in ids:
        p=source_by_id[idx];rows=[json.loads(x) for x in p.read_text().splitlines()]
        request=next(r for r in rows if r['kind']=='llm_input');frame=next(r for r in rows if r['kind']=='state' and r['step']==request['step'])
        inputs.append(dict(scenario_index=idx,source_trace=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),request=request,frame=frame))
    out.mkdir(exist_ok=False)
    save(out/'launch.json',dict(arms=['v4','priority_v5'],planned=20,states=ids,preferred_model='kimi-k3',
        approved_free_pool=approved,physical_budget_per_arm_scene=4,shared_bailian_capabilities='observed_glm5.3_true',
        source_sha256=hashes,source_model_before=before_model,started_at=time.time(),
        scope='frozen initial-state planning only; route-as-lanes proxy; no diffusion/control/closed-loop coverage claims'))
    save(out/'inputs.json',inputs)
    records=[];restore=install_provider_compatibility();fatal=False
    try:
        for number,item in enumerate(inputs):
            order=['v4','priority_v5'] if number%2==0 else ['priority_v5','v4']
            for arm in order:
                current=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
                active=[m for m in approved if m not in current['disabled_models']]
                pool=(['kimi-k3'] if 'kimi-k3' in active else [])+[m for m in active if m!='kimi-k3']
                if not pool:fatal=True;break
                os.environ['LLM_MODEL_NAME']=pool[0];os.environ['LLM_MODEL_NAMES']=','.join(pool)
                cfg=OmegaConf.load(ROOT/'experiments/skills_modelpool_recovery_development10_paired_20261023_r3/skills_full_shared_recovery/resolved_config.yaml')
                cfg.sim.llm.skills.root=str(HERE/'skills' if arm=='priority_v5' else ROOT/'work/skills_v2_compact_dev_20261020/skills')
                request=copy.deepcopy(item['request']);frame=item['frame'];state=request['state'];route=np.asarray(state['route'],float)
                env=NS(ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
                    agent_active=np.asarray(frame['active']),current_step=request['step'],current_scene_id=f"scenario_{item['scenario_index']:03d}",
                    scenario_dict={'route':route,'lanes':[route]},get_static_obstacles=lambda:[],cfg=cfg,dt=float(cfg.sim.dt),pending_joint_trajectory=None)
                env.get_state_for_planning=lambda state=state:state
                planner=LLMAdversarialPlanner(model_name=pool[0],model_names=pool,provider='dashscope',attack_mode='trajectory_only')
                if not planner.available:raise ValueError('bailian_unavailable')
                planner.client=planner.client.with_options(max_retries=0)
                calls=[];original_create=planner.client.chat.completions.create
                def create(*args,**kwargs):
                    if len(calls)>=4:raise RuntimeError('frozen_pair_physical_budget_exhausted')
                    event=dict(arm=arm,scenario_index=item['scenario_index'],model_requested=kwargs.get('model'),messages=kwargs.get('messages'),extra_body=kwargs.get('extra_body'));calls.append(event);begin=time.monotonic()
                    try:
                        response=original_create(*args,**kwargs);event.update(model_returned=response.model,content=response.choices[0].message.content,usage=response.usage.model_dump() if response.usage else None);return response
                    except Exception as error:
                        event.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None));raise
                    finally:
                        event['elapsed_seconds']=time.monotonic()-begin
                        with (out/'api_calls.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(event,ensure_ascii=False)+'\n')
                planner.client.chat.completions.create=create
                engine=(PrioritySkillsPlanner if arm=='priority_v5' else RouteTailReadyPlanner)(cfg.sim,4)
                start=time.monotonic();plan=engine.plan(planner,env,state,request.get('instruction') or ['使用动态离散攻击挑战自车，禁止静态障碍物。'])
                context=planner.last_trace.get('context',{});catalog=context.get('candidates',[])
                record=dict(arm=arm,scenario_index=item['scenario_index'],step=request['step'],plan=plan,trace=planner.last_trace,
                    accepted=bool(plan and plan.get('attack')),published_nonempty=bool(catalog),
                    semantics_nonempty_before_readiness=bool(catalog or context.get('preflight_withheld_candidates') or context.get('route_tail_withheld_candidates')),
                    physical_requests=len(calls),planner_service_failure=planner.last_request_failed,
                    elapsed_seconds=time.monotonic()-start,actual_attack_frames=0,D=None)
                records.append(record);save(out/f"scene_{item['scenario_index']:03d}_{arm}.json",record)
                save(out/'results.partial.json',dict(planned=20,attempted=len(records),records=records,unattempted=20-len(records),actual_execution_frames=0))
                if planner.last_request_failed:fatal=True;break
            if fatal:break
    finally:restore()
    source_ok=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    pool_ok=verify_pool_transition(before_model,inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'))
    result=dict(planned=20,attempted=len(records),unattempted=20-len(records),records=records,
        final_planner_service_failure=fatal,source_unchanged=source_ok,model_transition=pool_ok,
        actual_execution_frames=0,D=None,scope='planning only; no claim of effective attacks',goal_complete=False)
    save(out/'results.json',result);save(out/'exit_status.json',dict(exit_code=0 if len(records)==20 and not fatal and source_ok and pool_ok['verified'] else 1,finished_at=time.time()))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))


if __name__=='__main__':main()
