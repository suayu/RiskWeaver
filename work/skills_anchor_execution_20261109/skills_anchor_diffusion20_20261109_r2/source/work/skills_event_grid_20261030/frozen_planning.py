"""真实百炼冻结初态配对：只评估高层计划，原验证器及4次物理预算保留。"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,copy,hashlib,json,os,shutil,sys,time,uuid
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'work/skills_provider_compat_20261028'),str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from event_planner import FutureEventPlanner
from temporal_planner import TemporalCandidatePlanner
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from physical_call_budget import PhysicalCallBudget,bounded_model_request
from provider_control import install_control
from launch_pool_pair import verify_sources

FLAGS=dict(extra_recipes=True,rolling_intent=True,explicit_contract=True,deduplicate=True,fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
INSTRUCTION=['通过背景交通参与者的攻击性行为挑战自车；仅选择离散策略、目标和稀疏锚点。自车未来仅为假设，不能声称尚未执行的事件已经发生。']


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def engine_for(mode,cfg,sample):
    engine=TemporalCandidatePlanner(cfg.sim,4,**FLAGS) if mode=='temporal_original' else FutureEventPlanner(cfg.sim,4,event_grid=True,**FLAGS)
    context=copy.deepcopy(sample['contexts'][mode]);engine.priority_evidence=context['priority_evidence'];engine.incremental_cards=context['incremental_cards']
    engine.forecast=context['ego_future_reference'];engine.previous_status=context['rolling_intent']['previous']
    if mode=='full_grid':engine.event_grid_audit=sample['event_grid_audit']
    # 冻结状态实验明确重放已经构造的目录；仍通过原真实提示链、编译器和最终语义验证。
    engine.build_context=lambda *args,**kwargs:copy.deepcopy(context)
    return engine


def run_trial(sample,mode,pool,cfg,out,planner_factory=LLMAdversarialPlanner):
    out.mkdir(exist_ok=False);engine=engine_for(mode,cfg,sample)
    planner=planner_factory(model_name=pool[0],model_names=pool,provider='dashscope')
    if planner.provider!='dashscope' or not planner.available:raise RuntimeError('approved_bailian_client_unavailable')
    planner.use_obstacles=False;planner.allow_trajectory=True;planner.use_multimodal=False
    create=planner.client.chat.completions.create;journal=out/'api_calls.jsonl'
    def recorded(*args,**kwargs):
        start=time.monotonic();row=dict(physical_request_id=uuid.uuid4().hex,model_requested=kwargs.get('model'),messages=kwargs.get('messages'),
            response_format=kwargs.get('response_format'),extra_body=kwargs.get('extra_body'),started_at=time.time())
        try:
            response=create(*args,**kwargs);row.update(model_returned=response.model,content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else None,finish_reason=response.choices[0].finish_reason);return response
        except Exception as error:
            row.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None));raise
        finally:
            row['elapsed_seconds']=time.monotonic()-start
            with journal.open('a',encoding='utf-8') as file:file.write(json.dumps(row,ensure_ascii=False)+'\n')
    planner.client.chat.completions.create=recorded
    budget=PhysicalCallBudget(4,out/'budget_events.jsonl');budget.reset(sample['scene']);budget.attach(planner)
    original_request=LLMAdversarialPlanner._request_attack_plan
    planner._request_attack_plan=lambda *args,**kwargs:bounded_model_request(planner,original_request,*args,**kwargs)
    state=copy.deepcopy(sample['state']);frame=sample['frame']
    env=NS(current_step=0,current_scene_id=sample['scene_id'],dt=.1,agent_active=np.asarray(frame['active']),
        ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},get_state_for_planning=lambda:state)
    started=time.monotonic();result=dict(scene=sample['scene'],mode=mode,planned=True,attempted=True,accepted_plan=False,
        state_source_sha256=sample['source_trace_sha256'],new_execution_frames=0,D=None,semantic_review='not_reviewed',
        scope='initial_state_only_actual_highlevel_prompt_and_compile; not_executed_or_full_episode_coverage')
    try:
        plan=engine.plan(planner,env,state,INSTRUCTION)
        result.update(plan=plan,accepted_plan=bool(plan and plan.get('attack') is True),
            planner_service_failure=planner.last_request_failed,validation_reason=(planner.last_trace or {}).get('validation_reason'),
            planner_trace=planner.last_trace,selected_candidate=(planner.last_trace or {}).get('selected_candidate'),logical_requests=engine.calls)
    except Exception as error:
        result.update(runtime_failure=True,error_type=type(error).__name__,planner_service_failure=bool(planner._is_service_failure(error)))
    finally:
        result.update(budget=budget.snapshot(),elapsed_seconds=time.monotonic()-started)
        save(out/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--free-protection-confirmed',action='store_true');a=p.parse_args()
    if not a.free_protection_confirmed or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('confirmed_free_CPU_protocol_required')
    protocol=json.loads((a.protocol/'protocol.json').read_text());hashes=json.loads((a.protocol/'source_sha256.json').read_text())
    if not verify_sources(protocol,hashes)['verified']:raise ValueError('source_drift_before_planning_pair')
    if not (ROOT/protocol['free_protection_receipt']).is_file():raise ValueError('free_protection_receipt_missing')
    inputs=a.protocol/'inputs.json'
    if hashlib.sha256(inputs.read_bytes()).hexdigest()!=protocol['inputs_sha256']:raise ValueError('input_hash_changed')
    samples=json.loads(inputs.read_text())
    if len(samples)!=10 or [s['scene'] for s in samples]!=protocol['scenes']:raise ValueError('frozen_ten_scene_denominator_changed')
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(a.protocol/'resolved_config.yaml');a.output.mkdir(exist_ok=False)
    for name in ['protocol.json','inputs.json','resolved_config.yaml','source_sha256.json']:shutil.copyfile(a.protocol/name,a.output/name)
    shutil.copytree(a.protocol/'source',a.output/'source_snapshot')
    a.output.with_suffix('.pid').write_text(str(os.getpid()))
    save(a.output/'launch.json',dict(pid=os.getpid(),planned_state_arms=20,planned_scenes=protocol['scenes'],
        physical_calls_per_state_arm=4,maximum_total_physical_calls=80,actual_GPU_work=False,started_at=time.time()))
    rows=[];checks=[];restore=install_control('service_only')
    try:
        for index,sample in enumerate(samples):
            for mode in (['temporal_original','full_grid'] if index%2==0 else ['full_grid','temporal_original']):
                check=verify_sources(protocol,hashes);checks.append(check)
                save(a.output/'runtime_source_integrity.json',checks)
                if not check['verified']:raise ValueError('algorithm_drift_during_planning_pair')
                result=run_trial(sample,mode,protocol['frozen_model_pool'],cfg,a.output/f'scene{sample["scene"]:03d}_{mode}')
                rows.append(result);save(a.output/'results.partial.json',dict(planned_state_arms=20,results=rows))
                if result.get('runtime_failure') or result.get('planner_service_failure'):raise RuntimeError('final_planning_service_or_runner_failure')
        save(a.output/'results.json',dict(status='completed',planned_state_arms=20,results=rows,final_source_check=verify_sources(protocol,hashes),
            new_execution_frames=0,D=None,majority_correct_planning_certified=False,goal_complete=False))
    except Exception as error:
        save(a.output/'results.json',dict(status='failed',error_type=type(error).__name__,planned_state_arms=20,results=rows,
            unattempted_state_arms=20-len(rows),new_execution_frames=0,D=None,goal_complete=False));raise
    finally:restore()


if __name__=='__main__':main()
