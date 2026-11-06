"""真实百炼两次规划与无空间负例；合成道路集成冒烟，不代替真实场景覆盖实验。"""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,hashlib,json,os,sys,time,uuid
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(HERE),str(ROOT/'work/skills_modelpool_budget_dev_20261022'),str(ROOT/'work/skills_provider_compat_20261028')]
from test_ego_profile import environment
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.skills_planner import SkillsAttackPlanner
from policies.rag_store import AttackRAGStore
from policies.rag_runtime import AttackRAGRuntime
from hybrid_planner import DynamicFirstStaticPlanner
from physical_call_budget import PhysicalCallBudget,bounded_model_request
from provider_control import install_control
from model_pool_protocol import inspect_model_source
from launch_pool_pair import verify_sources

OUTPUT=ROOT/'experiments/skills_dynamic_static_highlevel_probe3_20261101_r1'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only')
    if not (ROOT/'work/bailian_pool_protection_confirmed_20261023.json').is_file():raise ValueError('free_protection_missing')
    OUTPUT.mkdir(exist_ok=False);OUTPUT.with_suffix('.pid').write_text(str(os.getpid()))
    model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    pool=[m for m in ['glm-5.3','qwen3.8-2.4t-a95b'] if m in model_source['active_models']]
    if not pool:raise ValueError('approved_healthy_pool_empty')
    names=set(json.loads((ROOT/'work/skills_source_facts_protocol_20261101_r4/source_sha256.json').read_text()))
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*') if p.is_file() and p.suffix in ('.py','.md','.json'))
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(names)}
    protocol=dict(version='dynamic_static_probe3_v1',planned=3,model_source=model_source,models=pool,
        physical_calls_per_case=4,maximum_physical_calls=12,timeout_seconds=180,max_tokens=8192,reasoning_effort='low',
        scope='synthetic_road_highlevel_integration_only_not_generalization_or_actual_attack',new_execution_frames=0,D=None,
        official_parameter_source='https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions',goal_complete=False)
    save(OUTPUT/'protocol.json',protocol);save(OUTPUT/'source_sha256.json',hashes)
    for name in hashes:
        target=OUTPUT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    save(OUTPUT/'launch.json',dict(pid=os.getpid(),started_at=time.time(),planned=3,CUDA_visible_devices='',new_execution_frames=0))
    rows=[];checks=[];restore=install_control('service_only')
    try:
        for index,kind in enumerate(['dynamic_vehicle','no_vehicle_static','no_forward_space']):
            output=OUTPUT/kind;output.mkdir();env=environment()
            if kind!='dynamic_vehicle':env.agent_active[:]=False
            if kind=='no_forward_space':
                route=np.array([[0.,0.],[5.,0.]])
                env.scenario_dict=dict(route=route,lanes=np.array([route]))
            original_state=env.get_state_for_planning
            env.get_state_for_planning=lambda:dict(original_state(),agents=original_state()['agents'] if env.agent_active.any() else [],
                                                  route=env.scenario_dict['route'].tolist())
            sim=NS(ego_profile=NS(),llm=NS(skills=NS(enabled=True,max_repairs=1,source_fact_contract_enabled=True)))
            engine=DynamicFirstStaticPlanner(SkillsAttackPlanner(sim,4),sim,4)
            planner=LLMAdversarialPlanner(model_name=pool[0],model_names=pool,provider='dashscope')
            if not planner.available:raise RuntimeError('Bailian_client_unavailable')
            planner.use_obstacles=True;planner.allow_trajectory=True;planner.use_multimodal=False
            planner.client=planner.client.with_options(timeout=180,max_retries=0)
            create=planner.client.chat.completions.create
            def recorded(*args,**kwargs):
                values=dict(kwargs,max_tokens=8192,reasoning_effort='low')
                values['extra_body']=dict(values.get('extra_body') or {},enable_thinking=True)
                row=dict(physical_request_id=uuid.uuid4().hex,model_requested=values['model'],messages=values['messages'],
                    extra_body=values['extra_body'],max_tokens=8192,reasoning_effort='low',timeout_seconds=180)
                started=time.monotonic()
                try:
                    response=create(*args,**values)
                    row.update(model_returned=response.model,content=response.choices[0].message.content,
                        finish_reason=response.choices[0].finish_reason,
                        usage=response.usage.model_dump() if response.usage else None)
                    return response
                except Exception as error:
                    row.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None));raise
                finally:
                    row['elapsed_seconds']=time.monotonic()-started
                    with (output/'api_calls.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(row,ensure_ascii=False)+'\n')
            planner.client.chat.completions.create=recorded
            budget=PhysicalCallBudget(4,output/'budget_events.jsonl');budget.reset(index);budget.attach(planner)
            original_request=LLMAdversarialPlanner._request_attack_plan
            planner._request_attack_plan=lambda prompt,scene_image=None:bounded_model_request(planner,original_request,prompt,scene_image=scene_image)
            check=verify_sources(protocol,hashes);checks.append(check);save(OUTPUT/'runtime_source_integrity.json',checks)
            if not check['verified']:raise ValueError('source_changed')
            plan=engine.plan(planner,env,env.get_state_for_planning(),['若有可行车辆攻击则优先使用车辆；否则从静态组合目录选择障碍物，挑战自车。'])
            row=dict(case=kind,plan=plan,planner_trace=planner.last_trace,
                highlevel_accepted=bool(plan and (plan.get('attack') or plan.get('obstacle_plan'))),
                planner_service_failure=planner.last_request_failed,budget=budget.snapshot(),
                new_execution_frames=0,D=None,collision=None,background_collision=None,prose_correctness='not_reviewed')
            save(output/'result.json',row);rows.append(row);save(OUTPUT/'results.partial.json',dict(planned=3,results=rows))
            if planner.last_request_failed:raise RuntimeError('final_service_failure')
        save(OUTPUT/'results.json',dict(status='completed',planned=3,results=rows,
            final_source_check=verify_sources(protocol,hashes),new_execution_frames=0,D=None,goal_complete=False))
    except Exception as error:
        save(OUTPUT/'results.json',dict(status='failed',error_type=type(error).__name__,planned=3,results=rows,
            unattempted=3-len(rows),new_execution_frames=0,D=None,goal_complete=False));raise
    finally:restore()


if __name__=='__main__':main()
