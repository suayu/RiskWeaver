"""固定目录/状态的真实百炼结构化事实对照；零仿真，不替代闭环攻击验收。"""
import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import time
import uuid

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_scene_facts_prototype_20261028'),
    str(ROOT/'work/skills_provider_compat_20261028'),str(ROOT/'work/skills_incremental_evidence_20261025'),
    str(ROOT/'work/skills_priority_v5_20261024'),str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from scene_facts import build_cards
from fact_planner import SceneFactPlanner,render_fact_prompt
from incremental_planner import IncrementalSkillsPlanner,render_incremental_prompt
from priority_planner import render_priority_prompt
from policies.intent_validation import canonical_hash
from policies.llm_adversarial_planner import LLMAdversarialPlanner,ModelOutputError
from provider_control import install_control
from physical_call_budget import PhysicalCallBudget,bounded_model_request,TransportBudgetExhausted
from launch_pool_pair import verify_sources


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def make_view(sample,enabled,request_id,feedback=None):
    payload=copy.deepcopy(sample['payload'])
    context=payload['context']
    if enabled:
        context['scene_fact_cards']=build_cards(sample['state'],context['candidates'],sample['source_step'])
        context['scene_fact_validation_scope']='typed_source_claims_only; raw_prose_not_certified; no_new_safety_or_distance_gate'
    payload.update(request_id=request_id,state_hash=canonical_hash(sample['state']),catalog_hash=canonical_hash(context),repair_feedback=feedback)
    raw='original base contract\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False)
    prompt=render_incremental_prompt(render_priority_prompt(raw))
    if enabled:prompt=render_fact_prompt(prompt)
    return prompt,payload


def validator_for(payload,enabled):
    engine=object.__new__(SceneFactPlanner if enabled else IncrementalSkillsPlanner)
    engine.priority_evidence=payload['context']['priority_evidence']
    engine.incremental_cards=payload['context']['incremental_cards']
    if enabled:
        engine.fact_grounding=True;engine.scene_fact_cards=payload['context']['scene_fact_cards']
    return engine


def run_trial(sample,enabled,pool,output):
    output.mkdir(exist_ok=False)
    planner=LLMAdversarialPlanner(model_name=pool[0],model_names=pool,provider='dashscope')
    if not planner.available:raise RuntimeError('bailian_client_unavailable')
    planner.last_trace={};create=planner.client.chat.completions.create
    journal=output/'api_calls.jsonl'
    def recorded(*args,**kwargs):
        start=time.monotonic()
        row=dict(physical_request_id=uuid.uuid4().hex,model_requested=kwargs.get('model'),
            messages=kwargs.get('messages'),response_format=kwargs.get('response_format'),extra_body=kwargs.get('extra_body'),started_at=time.time())
        try:
            response=create(*args,**kwargs);message=response.choices[0].message
            row.update(model_returned=response.model,content=message.content,
                usage=response.usage.model_dump() if response.usage else None,finish_reason=response.choices[0].finish_reason)
            return response
        except Exception as error:
            row.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None));raise
        finally:
            row['elapsed_seconds']=time.monotonic()-start
            with journal.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    planner.client.chat.completions.create=recorded
    budget=PhysicalCallBudget(4,output/'budget_events.jsonl');budget.reset(sample['scenario_index']);budget.attach(planner)
    request_id=uuid.uuid4().hex;feedback=None;attempts=[]
    result=dict(scenario_index=sample['scenario_index'],source_step=sample['source_step'],fact_grounding=enabled,
        schema_valid=False,ranking=None,attempts=attempts,actual_attack_frames=0,D=None,
        planner_service_failure=False,budget_exhausted=False,
        scope='frozen-state_scene_and_ranking_contract_only_not_executed_attack',started_at=time.time())
    try:
        # 一次原请求加最多一次修复；故障回退共用4次实际SDK预算。
        for repair in range(2):
            prompt,payload=make_view(sample,enabled,request_id,feedback)
            engine=validator_for(payload,enabled)
            record=dict(repair=repair,prompt=prompt,request_id=request_id,state_hash=payload['state_hash'],catalog_hash=payload['catalog_hash'])
            attempts.append(record)
            try:
                _,response=bounded_model_request(planner,LLMAdversarialPlanner._request_attack_plan,prompt)
            except ModelOutputError:
                record['validation']='invalid_json_output';feedback=dict(reason='invalid_json_output')
                continue
            record['response']=response
            try:
                catalog={c['candidate_id']:c for c in payload['context']['candidates']}
                engine.validate_response(response,request_id,payload['state_hash'],payload['catalog_hash'],catalog,False)
            except (ValueError,TypeError) as error:
                record['validation']=str(error);feedback=dict(reason=str(error),previous_response=response)
                continue
            record['validation']='accepted_contract_only'
            result.update(schema_valid=True,ranking=response['ranking'],has_ranked_intent=bool(response['ranking']))
            break
    except Exception as error:
        result.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None),
            planner_service_failure=bool(planner._is_service_failure(error)),budget_exhausted=isinstance(error,TransportBudgetExhausted))
    finally:
        result.update(budget=budget.snapshot(),logical_queries=len(attempts),finished_at=time.time())
        save(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--free-protection-confirmed',action='store_true');args=p.parse_args()
    if not args.free_protection_confirmed:raise ValueError('free_protection_receipt_required')
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_diagnostic_requires_hidden_GPUs')
    protocol=json.loads((args.protocol/'protocol.json').read_text())
    hashes=json.loads((args.protocol/'source_sha256.json').read_text())
    check=verify_sources(protocol,hashes)
    if not check['verified']:raise ValueError('source_drift_before_frozen_diagnostic')
    if not (ROOT/protocol['free_protection_receipt']).exists():raise ValueError('missing_confirmed_protection')
    inputs=args.protocol/'inputs.json'
    if hashlib.sha256(inputs.read_bytes()).hexdigest()!=protocol['inputs_sha256']:raise ValueError('input_drift')
    samples=json.loads(inputs.read_text());args.output.mkdir(exist_ok=False)
    for name in ['inputs.json','protocol.json','source_sha256.json']:
        shutil.copyfile(args.protocol/name,args.output/name)
    shutil.copytree(args.protocol/'source',args.output/'source_snapshot')
    save(args.output/'launch.json',dict(pid=os.getpid(),planned_attempts=8,max_physical_requests_per_attempt=4,
        max_physical_requests_total=32,no_GPU=True,protocol=str(args.protocol),source_verified=check,started_at=time.time()))
    results=[];restore=install_control('service_only')
    try:
        for i,sample in enumerate(samples):
            # 交替两臂先后顺序，全部8预定保留；不按历史表现提前终止或多次重试。
            for enabled in ([False,True] if i%2==0 else [True,False]):
                label=sample['case_id']+('_facts_on' if enabled else '_facts_off')
                current=verify_sources(protocol,hashes)
                if not current['verified']:raise ValueError('algorithm_drift_during_frozen_diagnostic')
                result=run_trial(sample,enabled,protocol['frozen_model_pool'],args.output/label)
                results.append(dict(trial=label,**result));save(args.output/'results.partial.json',dict(planned_attempts=8,results=results))
        save(args.output/'results.json',dict(status='completed',planned_attempts=8,results=results,
            final_source_check=verify_sources(protocol,hashes),new_execution_frames=0,D=None,goal_complete=False))
    except Exception as error:
        save(args.output/'results.json',dict(status='runner_failure',error_type=type(error).__name__,planned_attempts=8,results=results,
            unattempted=8-len(results),new_execution_frames=0,D=None,goal_complete=False))
        raise
    finally:restore()


if __name__=='__main__':
    main()
