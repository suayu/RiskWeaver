"""只读比较实际攻击和攻击前恒速参考；不是无攻击闭环因果实验或新的安全门控。"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np


def box(state):
    a=np.asarray(state,dtype=float)
    if len(a)<7 or not np.isfinite(a[:7]).all() or min(a[5:7])<=0:
        raise ValueError('invalid_OBB_state')
    c,s=math.cos(a[4]),math.sin(a[4])
    axes=np.array([[c,s],[-s,c]])
    return np.array([a[:2]+sx*a[5]/2*axes[0]+sy*a[6]/2*axes[1]
                     for sx,sy in [(1,1),(-1,1),(-1,-1),(1,-1)]])


def point_segment(point,a,b):
    d=b-a;den=float(d@d)
    fraction=float(np.clip((point-a)@d/den,0.,1.)) if den else 0.
    return float(np.linalg.norm(point-(a+fraction*d)))


def clearance(first,second):
    """矩形相交返回0，否则返回欧氏边界距离；仅是所给离散时刻的OBB距离。"""
    a,b=box(first),box(second)
    separated=False
    for polygon in [a,b]:
        for k in [0,1]:
            edge=polygon[(k+1)%4]-polygon[k]
            axis=np.array([-edge[1],edge[0]])
            pa,pb=a@axis,b@axis
            if pa.max()<pb.min()-1e-10 or pb.max()<pa.min()-1e-10:
                separated=True
    if not separated:return 0.
    return min(point_segment(p,q[k],q[(k+1)%4]) for pset,q in [(a,b),(b,a)] for p in pset for k in range(4))


def cv(state,seconds):
    a=np.asarray(state,dtype=float).copy();a[:2]+=a[2:4]*seconds
    return a


def execution_ledger(rows):
    """兼容旧计划缺request_id：必须联合目标与时间，不能把多个null计划合并。"""
    outputs={r['step']:r.get('validated_plan') or {} for r in rows if r.get('kind')=='llm_output'}
    intents={r['step']:r.get('attack_intent') or {} for r in rows if r.get('kind')=='prediction'}
    forecasts={r['step']:(r.get('joint') or {}).get('metadata',{}) for r in rows if r.get('kind')=='execution_prediction'}
    plans=[];unmatched=[];unguided=[];markers=[]
    for r in rows:
        if r.get('kind')=='plan_accepted':
            p=outputs.get(r['step'],{});intent=intents.get(r['step'],{})
            plans.append(dict(source_step=r['step'],target_id=r['target_id'],request_id=r.get('request_id'),
                duration_frames=int(intent.get('duration',p.get('duration',10))),
                strategy=p.get('strategy'),candidate_id=p.get('candidate_id'),anchors=p.get('anchors',[]),
                attribution='request_id_target_and_time' if r.get('request_id') else 'legacy_target_and_time',
                actual_execution_steps=[]))
    for r in rows:
        if r.get('kind')!='attack_execution' or not r.get('executed'):continue
        source=r.get('diffusion_source_step',r['step']-1);markers.append(r['step'])
        possible=[p for p in plans if p['source_step']<=source and p['target_id']==r['target_id']
                  and (r.get('request_id') is None or r['request_id']==p['request_id'])]
        if not possible:unmatched.append(r['step']);continue
        chosen=max(possible,key=lambda p:p['source_step'])
        chosen['actual_execution_steps'].append(r['step'])
        if not forecasts.get(source,{}).get('anchor_guidance_active'):unguided.append(r['step'])
    return dict(plans=plans,unmatched=unmatched,unguided=unguided,duplicate_markers=len(markers)-len(set(markers)))


def measure_plan(plan,states,dt=.1):
    """完整有效期内对比；缺状态、到期后帧、缺活跃目标均保持无效，不补0。"""
    result=dict(plan,valid=False,reason=None,increment_fixed_ego_CV_OBB_m=None,
                increment_observed_ego_OBB_m=None,increment_fixed_ego_CV_center_m=None)
    start=plan['source_step'];duration=plan['duration_frames'];target=plan['target_id']
    expected=list(range(start+1,start+duration+1))
    if plan['actual_execution_steps']!=expected:
        result['reason']='not_full_declared_guided_window';return result
    if any(s not in states for s in range(start,start+duration+1)):
        result['reason']='missing_state';return result
    selected=[states[s] for s in range(start,start+duration+1)]
    if any(isinstance(target,bool) or target<0 or target>=len(s['agents']) or target>=len(s['active']) or not s['active'][target] for s in selected):
        result['reason']='inactive_or_missing_target';return result
    try:
        initial=selected[0];ego0=initial['ego'];actor0=initial['agents'][target]
        # 同一时刻分别比较实际与CV目标；观察ego路径本身受攻击影响，故不作因果解释。
        curve=[]
        for i,state in enumerate(selected):
            t=i*dt;ego_ref=cv(ego0,t);target_ref=cv(actor0,t)
            ego_actual=state['ego'];target_actual=state['agents'][target]
            curve.append(dict(seconds=t,
                actual_target_to_reference_ego_OBB_m=clearance(target_actual,ego_ref),
                reference_target_to_reference_ego_OBB_m=clearance(target_ref,ego_ref),
                actual_target_to_observed_ego_OBB_m=clearance(target_actual,ego_actual),
                reference_target_to_observed_ego_OBB_m=clearance(target_ref,ego_actual),
                actual_target_to_reference_ego_center_m=float(np.linalg.norm(np.asarray(target_actual[:2])-ego_ref[:2])),
                reference_target_to_reference_ego_center_m=float(np.linalg.norm(target_ref[:2]-ego_ref[:2]))))
    except (ValueError,TypeError,IndexError) as error:
        result['reason']='invalid_geometry:'+type(error).__name__;return result
    fields=[k for k in curve[0] if k!='seconds']
    minima={k:min(r[k] for r in curve) for k in fields}
    result.update(valid=True,curve=curve,minima=minima,
        increment_fixed_ego_CV_OBB_m=minima['reference_target_to_reference_ego_OBB_m']-minima['actual_target_to_reference_ego_OBB_m'],
        increment_observed_ego_OBB_m=minima['reference_target_to_observed_ego_OBB_m']-minima['actual_target_to_observed_ego_OBB_m'],
        increment_fixed_ego_CV_center_m=minima['reference_target_to_reference_ego_center_m']-minima['actual_target_to_reference_ego_center_m'])
    return result


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit_experiment(root,arms):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected=manifest['original_scenario_indices'];hashes={str(root/'manifest.json'):digest(root/'manifest.json')}
    groups={}
    for arm in arms:
        attempts=[];seen=set()
        for path in sorted((root/arm).rglob('attempt_result.json')):
            a=json.loads(path.read_text(encoding='utf-8'));index=a['scenario_index']
            if index in seen or index not in expected:raise ValueError('duplicate_or_unknown_attempt')
            seen.add(index);trace=path.with_name('execution_trace.jsonl')
            hashes[str(path)]=digest(path);hashes[str(trace)]=digest(trace)
            rows=[json.loads(x) for x in trace.read_text(encoding='utf-8').splitlines()]
            states={r['step']:r for r in rows if r.get('kind')=='state'}
            ledger=execution_ledger(rows)
            if ledger['unmatched'] or ledger['unguided'] or ledger['duplicate_markers']:
                raise ValueError('execution_attribution_invalid')
            plans=[measure_plan(p,states) for p in ledger['plans']]
            # 原统计必须可追溯，不能在衍生分析中静默丢掉未执行计划。
            if len(plans)!=a['attack_plan_count'] or sum(len(p['actual_execution_steps']) for p in plans)!=a['attack_executed_frames']:
                raise ValueError('plan_or_execution_count_mismatch')
            attempts.append(dict(scenario_index=index,attempted=True,executed_steps=a['executed_steps'],
                generation_failure=a['generation_failure'],runtime_failure=a['runtime_failure'],
                service_failures=a['planner_service_failure_count'],D=a['scenario_danger_score'],
                plans=plans,attack_frames=a['attack_executed_frames']))
        attempts.extend(dict(scenario_index=i,attempted=False,executed_steps=0,D=None,plans=[],attack_frames=0,
                             reason='unattempted_not_normal_no_attack') for i in expected if i not in seen)
        all_plans=[p for a in attempts for p in a['plans']];valid=[p for p in all_plans if p['valid']]
        positive=[p for p in valid if p['increment_fixed_ego_CV_OBB_m']>1e-6]
        groups[arm]=dict(planned_scenes=len(expected),attempted_scenes=len(seen),
            attack_scenes=sum(a['attack_frames']>0 for a in attempts),
            executed_attack_frames=sum(a['attack_frames'] for a in attempts),
            accepted_plans=len(all_plans),executed_plans=sum(bool(p['actual_execution_steps']) for p in all_plans),
            valid_full_window_plans=len(valid),invalid_plan_reasons=dict(Counter(p['reason'] for p in all_plans if not p['valid'])),
            positive_fixed_ego_CV_OBB_plans=len(positive),
            zero_fixed_ego_CV_OBB_plans=sum(abs(p['increment_fixed_ego_CV_OBB_m'])<=1e-6 for p in valid),
            negative_fixed_ego_CV_OBB_plans=sum(p['increment_fixed_ego_CV_OBB_m']< -1e-6 for p in valid),
            scene_with_positive_proxy_count=sum(any(p['valid'] and p['increment_fixed_ego_CV_OBB_m']>1e-6 for p in a['plans']) for a in attempts),
            median_fixed_ego_CV_OBB_difference_m=float(np.median([p['increment_fixed_ego_CV_OBB_m'] for p in valid])) if valid else None,
            median_observed_ego_OBB_difference_m=float(np.median([p['increment_observed_ego_OBB_m'] for p in valid])) if valid else None,
            attempts=attempts)
    if any(digest(Path(n))!=h for n,h in hashes.items()):raise RuntimeError('immutable_input_changed_during_audit')
    return dict(experiment=str(root),groups=groups,input_sha256=hashes,inputs_unchanged=True,
        measured_API_requests=0,new_execution_frames=0,changes_to_D_definition=False,
        reference_type='attacker_source_constant_velocity_with_fixed_yaw; ego_CV_or_observed_ego',
        limitations=['not_a_closed_loop_no_attack_counterfactual','observed_ego_is_post_treatment',
                     'CV_invalid_on_curved_road_or_reacting_driver','discrete_0.1s_OBB_distance_not_continuous_clearance',
                     'difference_of_minima_may_occur_at_different_times','positive_proxy_not_correct_strategy_or_threat_proof'],goal_complete=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--arms',nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.experiment.resolve() in args.output.resolve().parents:
        raise ValueError('preserve_existing_experiment_and_audit')
    result=audit_experiment(args.experiment,args.arms)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{n:v for n,v in g.items() if n!='attempts'} for k,g in result['groups'].items()},ensure_ascii=False))


if __name__=='__main__':main()
