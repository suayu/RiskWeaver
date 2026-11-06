"""同一攻击的绝对时间对齐与跟随审计；仅重算历史事实，不修改轨迹细化。"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np


def sample(anchors,times,interval=1.):
    """四个稀疏锚点按原语义插值；超出预测时域拒绝，不能静默外推。"""
    a=np.asarray(anchors,float);t=np.asarray(times,float)
    if a.shape!=(4,2) or not np.isfinite(a).all() or not np.isfinite(t).all() or interval<=0:
        raise ValueError('invalid_sparse_intention')
    if np.any(t < -1e-10) or np.any(t > 3*interval+1e-10):raise ValueError('outside_sparse_intention_horizon')
    return np.column_stack([np.interp(np.clip(t,0,3*interval),np.arange(4)*interval,a[:,i]) for i in range(2)])


def continuity(prior,new,current_actor_position,dt=.1,interval=1.):
    """比较同一世界时刻；同时报告原始跳变与扣除起点跟随偏差后的形状跳变。"""
    if prior['target_id']!=new['target_id']:return dict(comparable=False,reason='different_target_new_episode')
    elapsed=(new['source_step']-prior['source_step'])*dt
    if elapsed<=0 or elapsed>=3*interval:return dict(comparable=False,reason='no_forward_overlap')
    overlap=3*interval-elapsed
    # 两条分段直线差值的范数为分段凸函数；公共折点能给出精确最大值。
    knots=sorted(set([0.,overlap]+[float(x) for x in np.arange(4)*interval if 0<x<overlap]
        +[float(x-elapsed) for x in np.arange(4)*interval if 0<x-elapsed<overlap]))
    old=sample(prior['anchors'],np.asarray(knots)+elapsed,interval);fresh=sample(new['anchors'],knots,interval)
    tracking_offset=np.asarray(current_actor_position,float)-old[0]
    raw=np.linalg.norm(fresh-old,axis=1);shape=np.linalg.norm(fresh-(old+tracking_offset),axis=1)
    return dict(comparable=True,elapsed_seconds=elapsed,overlap_seconds=overlap,
        evaluated_new_relative_times_seconds=knots,absolute_times_seconds=[new['source_step']*dt+t for t in knots],
        old_future_world=old.tolist(),new_future_world=fresh.tolist(),
        source_tracking_offset_m=tracking_offset.tolist(),source_tracking_error_m=float(np.linalg.norm(tracking_offset)),
        raw_overlap_deviations_m=raw.tolist(),tracking_aligned_shape_deviations_m=shape.tolist(),
        max_raw_overlap_deviation_m=float(raw.max()),max_tracking_aligned_shape_deviation_m=float(shape.max()),
        scope='forecast_intention_overlap; previous10frame_lease_has_expired; not_old_executable_control_reuse')


def tracking_metrics(plan,states,joint,dt=.1,interval=1.):
    """预测第0项是t+dt；只有实际10帧租期内的状态可用于该计划执行误差。"""
    target=plan['target_id'];step=plan['source_step'];actual=[]
    for i in range(1,plan['duration_frames']+1):
        state=states.get(step+i)
        if not state or target>=len(state['active']) or not state['active'][target]:
            return dict(valid=False,reason='missing_or_inactive_actual_state')
        reference=sample(plan['anchors'],[i*dt],interval)[0]
        position=np.asarray(state['agents'][target][:2],float)
        actual.append(dict(seconds=i*dt,position_world=position.tolist(),reference_world=reference.tolist(),error_m=float(np.linalg.norm(position-reference))))
    result=dict(valid=True,actual_declared_lease_samples=actual,actual_lease_end_error_m=actual[-1]['error_m'],
        actual_lease_RMSE_m=float(np.sqrt(np.mean([r['error_m']**2 for r in actual]))),
        predicted_anchor_errors=[],prediction_missing=False,later_actual_states_not_scored_against_expired_plan=True)
    if not joint or target not in joint.get('agent_ids',[]):result['prediction_missing']=True;return result
    row=joint['agent_ids'].index(target);positions=joint['positions_global'][row];valid=joint['valid_mask'][row]
    for seconds in [1.,2.,3.]:
        index=round(seconds/dt)-1
        if index>=len(positions) or not valid[index]:
            result['predicted_anchor_errors'].append(dict(seconds=seconds,error_m=None,reason='unavailable_prediction'));continue
        p=np.asarray(positions[index],float);reference=sample(plan['anchors'],[seconds],interval)[0]
        result['predicted_anchor_errors'].append(dict(seconds=seconds,prediction_index=index,prediction_world=p.tolist(),anchor_world=reference.tolist(),error_m=float(np.linalg.norm(p-reference))))
    result['prediction_first_sample_matches_next_actual']=bool(np.allclose(positions[0],actual[0]['position_world'],atol=1e-6))
    result['dynamics_projection_metadata']=(joint.get('metadata') or {}).get('dynamics_projection')
    return result


def audit(experiment):
    experiment=Path(experiment)
    if not (experiment/'launcher_result.json').exists():raise ValueError('terminal_historical_experiment_required')
    trace=experiment/'skills_full_shared_recovery/movies/scenario_074/execution_trace.jsonl';attempt=trace.with_name('attempt_result.json')
    watched={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [trace,attempt,experiment/'manifest.json']}
    rows=[json.loads(x) for x in trace.read_text(encoding='utf-8').splitlines()];states={r['step']:r for r in rows if r.get('kind')=='state'}
    joints={r['step']:r['joint'] for r in rows if r.get('kind')=='execution_prediction'}
    plans=[]
    for row in rows:
        p=row.get('validated_plan') or {}
        if row.get('kind')!='llm_output' or not p.get('attack'):continue
        step=row['step'];cid=p['candidate_id'];context=row['planner_trace']['context'];facts=context['priority_evidence'][cid]['facts']
        plan=dict(source_step=step,target_id=p['attack_target_id'],strategy=p['strategy'],candidate_id=cid,
            anchors=p['anchors'],duration_frames=p['duration'],ego_source=states[step]['ego'],actor_source=states[step]['agents'][p['attack_target_id']],
            active_window_fact=facts[cid+':active'],after_window_fact=facts[cid+':after'],contrast=context['incremental_cards'][cid],joint=joints.get(step))
        plan['tracking']=tracking_metrics(plan,states,joints.get(step));plans.append(plan)
    if [(p['source_step'],p['target_id']) for p in plans]!=[(0,0),(11,0),(55,1)]:raise ValueError('expected_specific_historical_three_plan_074')
    comparisons=[]
    for prior,new in zip(plans,plans[1:]):
        value=continuity(prior,new,new['actor_source'][:2]);value.update(old_source_step=prior['source_step'],new_source_step=new['source_step'])
        if not value['comparable']:value['prior_target_active_at_new_source']=bool(states[new['source_step']]['active'][prior['target_id']])
        comparisons.append(value)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in watched.items())
    return dict(version='source074_absolute_time_and_tracking_audit_20261028_v1',experiment=str(experiment),scenario_index=74,
        plans=plans,replans=comparisons,input_sha256=watched,inputs_unchanged=True,
        time_semantics=dict(dt_seconds=.1,sparse_anchor_times_seconds=[0,1,2,3],execution_lease_seconds=1.,observed_replan_source_steps=[0,11,55]),
        source_only_CV_prediction_not_true_future_or_actual_policy_forecast=True,
        no_claim_that_terminal_anchor_must_be_collision_point=True,new_API_requests=0,new_execution_frames=0,
        current_runtime_and_diffusion_unchanged=True,goal_complete=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=audit(args.experiment)
    if args.experiment.resolve()==args.output.resolve() or args.experiment.resolve() in args.output.resolve().parents:raise ValueError('do_not_write_experiment')
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(plans=len(result['plans']),replans=[{k:v for k,v in r.items() if k in ['comparable','reason','source_tracking_error_m','max_raw_overlap_deviation_m','max_tracking_aligned_shape_deviation_m']} for r in result['replans']],tracking=[dict(step=p['source_step'],lease_end_error=p['tracking']['actual_lease_end_error_m'],predicted_anchor_errors=[v['error_m'] for v in p['tracking']['predicted_anchor_errors']]) for p in result['plans']],new_API_requests=0),ensure_ascii=True))


if __name__=='__main__':main()
