"""只读已结束场景：请求时机、模型拒绝与连续执行分开，不写实验目录。"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(path):
    sources=[path,path.with_name('execution_trace.jsonl')]
    before={str(p):digest(p) for p in sources}
    attempt=json.loads(path.read_text())
    rows=[json.loads(line) for line in sources[1].read_text().splitlines()]
    states=[r for r in rows if r['kind']=='state']
    inputs={r['step']:r['state'] for r in rows if r['kind']=='llm_input'}
    outputs=[r for r in rows if r['kind']=='llm_output']
    markers=[r for r in rows if r['kind']=='attack_execution' and r.get('executed')]
    first_active=next((r['step'] for r in states if any(r.get('active',[]))),None)
    first_near=next((r['step'] for r in states if any(bool(on) and math.dist(a[:2],r['ego'][:2])<=30
        for a,on in zip(r['agents'],r.get('active',[])))),None)
    events=[]
    for output in outputs:
        step=output['step'];state=inputs.get(step);plan=output.get('validated_plan');trace=output.get('planner_trace') or {}
        visible=len(state['agents']) if state is not None else None
        category=('accepted_model_attack' if plan and plan.get('attack') else
                  'model_no_attack' if plan and not plan.get('attack') else 'rejected_or_skipped_plan')
        context=trace.get('context') or {}
        events.append(dict(step=step,visible_agents=visible,category=category,
            reason=(plan or {}).get('reason') or output.get('planner_skipped_reason') or trace.get('validation_reason'),
            planner_stage=trace.get('stage'),plan_validation=trace.get('plan_validation'),
            catalog_count=len(context.get('candidates',[])) if context else None,
            target_id=(plan or {}).get('attack_target_id'),anchors=(plan or {}).get('anchors'),
            request_failed=output.get('request_failed'),logical_calls=(max(0,int(trace.get('api_calls_total',0))-int(trace.get('api_calls_before',0)))
                if trace.get('mode')=='skills_candidate_ranking' else len(trace.get('attempted_models',[])))))
    after={str(p):digest(p) for p in sources}
    stable=before==after
    return dict(scenario_index=attempt['scenario_index'],status=attempt.get('status'),
        executed_steps=attempt.get('executed_steps'),generation_failure=attempt.get('generation_failure'),
        generation_failure_reason=attempt.get('generation_failure_reason'),
        plans_accepted=sum(r['kind']=='plan_accepted' for r in rows),
        actual_attack_steps=[r['step'] for r in markers],D=attempt.get('scenario_danger_score'),
        D_valid_frames=attempt.get('danger_valid_frame_count'),first_active_step=first_active,
        first_active_within_30m_step=first_near,queries=events,
        all_queries_before_any_active=bool(events) and first_active is not None and max(e['step'] for e in events)<first_active,
        query_limit_note='logical events; actual physical transport counts require independent API records',
        source_sha256_at_read=before,finished_sources_stable_during_read=stable)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--arm',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve()==args.experiment.resolve() or args.experiment.resolve() in args.output.resolve().parents:
        raise ValueError('analysis cannot write the experiment')
    manifest=json.loads((args.experiment/'manifest.json').read_text());expected=manifest['original_scenario_indices']
    scenes=[summarize(p) for p in sorted((args.experiment/args.arm).rglob('attempt_result.json'))]
    api=args.experiment/f'{args.arm}_api_calls.jsonl';calls=[]
    if api.exists():calls=[json.loads(x) for x in api.read_text().splitlines()]
    for s in scenes:
        subset=[a for a in calls if a.get('scenario_index')==s['scenario_index']]
        s['physical_requests']=len(subset)
        s['physical_requests_with_empty_query_state']=sum(next((q['visible_agents']==0 for q in s['queries'] if q['step']==a.get('source_step')),False) for a in subset)
        s['model_distribution']=dict(Counter(a.get('model_returned') for a in subset if a.get('model_returned')))
    attempted={s['scenario_index'] for s in scenes}
    result=dict(scope='partial snapshot of ended attempts only; not final paired acceptance or method comparison',
        experiment=str(args.experiment),arm=args.arm,planned_scenes=len(expected),ended_attempts=len(scenes),
        remaining_or_in_progress_indices=[i for i in expected if i not in attempted],
        all_finished_sources_stable=all(s['finished_sources_stable_during_read'] for s in scenes),
        scenes=scenes,physical_requests_in_ended_attempts=sum(s['physical_requests'] for s in scenes),
        empty_input_physical_requests=sum(s['physical_requests_with_empty_query_state'] for s in scenes),
        plans_accepted=sum(s['plans_accepted'] for s in scenes),
        attack_frames=sum(len(s['actual_attack_steps']) for s in scenes),
        generation_failure_reasons=dict(Counter(s['generation_failure_reason'] for s in scenes if s['generation_failure'])),
        inference='early budget depletion is a plausible opportunity-coverage bottleneck; waiting for new-arm evidence; no claim that every later active vehicle is feasible to attack')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='scenes'},ensure_ascii=False))


if __name__=='__main__':main()
