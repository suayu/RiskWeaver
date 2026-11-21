"""仅从当前快照构造未来参照与稀疏意图记忆；不是自车真实未来或连续控制器。"""
import copy
import hashlib
import math
import numpy as np
from temporal_audit import sample,continuity


def ego_hypotheses(state,step,dt=.1):
    """CV与路线恒速是有标记的假设，不读取回放中的未来状态，不冒充概率分布。"""
    ego=np.asarray(state['ego_state'],float)
    if ego.ndim!=1 or len(ego)<5 or not np.isfinite(ego[:5]).all():raise ValueError('invalid_current_ego')
    if not math.isfinite(dt) or dt<=0:raise ValueError('invalid_dt')
    times=np.linspace(0,3,31);speed=float(np.linalg.norm(ego[2:4]))
    result=dict(source_step=int(step),dt_seconds=float(dt),relative_times_seconds=times.tolist(),
        absolute_times_seconds=(step*dt+times).tolist(),hypotheses=[dict(id='source_CV',
            positions_world=(ego[:2]+times[:,None]*ego[2:4]).tolist(),basis='current_measured_position_velocity')],
        probabilities=None,uses_future_ground_truth=False,actual_policy_rollout_available=False,
        uncertainty_scope='finite_deterministic_hypotheses_not_calibrated_probability_or_ego_response',
        route_status='unavailable_or_incompatible')
    route=np.asarray(state.get('route',[]),float)
    if route.ndim!=2 or route.shape[1]<2 or len(route)<2 or not np.isfinite(route[:,:2]).all():return result
    points=route[:,:2];segments=np.diff(points,axis=0);lengths=np.linalg.norm(segments,axis=1)
    good=lengths>1e-8
    if not good.any():return result
    fractions=np.zeros(len(lengths));fractions[good]=np.clip(np.sum((ego[:2]-points[:-1][good])*segments[good],axis=1)/(lengths[good]**2),0,1)
    projections=points[:-1]+fractions[:,None]*segments
    distances=np.linalg.norm(projections-ego[:2],axis=1);distances[~good]=np.inf
    i=int(np.argmin(distances));direction=segments[i]/lengths[i]
    # 路线方向不匹配或当前不在附近时保留CV并明确缺失，不伪造弯道预测。
    if distances[i]>3 or speed<1e-6 or float(direction@ego[2:4]/speed)<.5:return result
    arclength=np.r_[0,np.cumsum(lengths)];s0=arclength[i]+fractions[i]*lengths[i]
    wanted=s0+times*speed
    if wanted[-1]>arclength[-1]+1e-8:
        result['route_status']='route_tail_shorter_than_forecast_no_extrapolation';return result
    future=np.column_stack([np.interp(wanted,arclength,points[:,k]) for k in range(2)])
    # 原始横向偏差仅作几何平移，使t0严格等于测量；该假设仍不认证动力学可行。
    future+=ego[:2]-projections[i]
    result['hypotheses'].append(dict(id='route_constant_speed',positions_world=future.tolist(),
        basis='current_speed_along_known_route_with_constant_source_offset',dynamics_certified=False))
    result['route_status']='covered_geometric_hypothesis';return result


def forecast_point(forecast,hypothesis,seconds):
    times=np.asarray(forecast['relative_times_seconds'],float)
    if not 0<=seconds<=times[-1]:raise ValueError('ego_forecast_extrapolation_forbidden')
    p=np.asarray(hypothesis['positions_world'],float)
    return np.array([np.interp(seconds,times,p[:,k]) for k in range(2)])


def future_card(candidate,forecast,dt=.1):
    """同时间比较攻击意图与每个自车假设；分开执行有效期和预测时域。"""
    seconds=np.asarray(forecast['relative_times_seconds']);intent=sample(candidate['anchors'],seconds)
    active=seconds<=candidate['duration']*dt+1e-9;rows=[]
    for h in forecast['hypotheses']:
        gaps=np.linalg.norm(intent-np.asarray(h['positions_world']),axis=1)
        a=np.where(active)[0][int(np.argmin(gaps[active]))];full=int(np.argmin(gaps))
        rows.append(dict(hypothesis=h['id'],active_min_center_gap_m=float(gaps[a]),active_min_time_seconds=float(seconds[a]),
            forecast_min_center_gap_m=float(gaps[full]),forecast_min_time_seconds=float(seconds[full]),
            forecast_min_outside_active_lease=bool(not active[full])))
    return dict(rows=rows,active_lease_seconds=float(candidate['duration']*dt),forecast_seconds=3.,
        worst_hypothesis_active_min_gap_m=max(r['active_min_center_gap_m'] for r in rows),
        center_distances_not_OBB_TTC_or_risk_or_inevitable_collision=True)


class AttackMemory:
    """缓存上次接收的离散意图；绝不声称其已经实际执行，不跨场景或失活车辆复用。"""
    def __init__(self):self.previous=None;self.serial=0

    def clear(self):self.previous=None

    def inspect(self,scene,step,active,dt=.1):
        p=self.previous
        if p is None:return dict(available=False,reason='no_previous_accepted_intent')
        if p['scene_id']!=str(scene):reason='scene_changed'
        elif step<p['source_step']:reason='source_time_reversed'
        elif p['target_id']>=len(active) or not active[p['target_id']]:reason='previous_target_inactive'
        elif (step-p['source_step'])*dt>=3:reason='previous_forecast_expired'
        else:return dict(available=True,previous=copy.deepcopy(p),
            elapsed_seconds=float((step-p['source_step'])*dt),
            previous_lease_expired=bool((step-p['source_step'])*dt>=p['duration_frames']*dt),
            accepted_intent_only_execution_unconfirmed=True)
        self.clear();return dict(available=False,reason=reason)

    def commit(self,scene,step,plan,forecast,previous_status,dt=.1):
        if not isinstance(plan,dict) or not plan.get('attack'):return dict(updated=False,reason='no_new_accepted_attack')
        prior=previous_status.get('previous');same=bool(prior and prior['target_id']==plan['attack_target_id'])
        self.serial+=not same
        episode=prior['episode_id'] if same else hashlib.sha256(f'{scene}:{step}:{plan["attack_target_id"]}:{self.serial}'.encode()).hexdigest()[:16]
        c=dict(anchors=plan['anchors'],duration=plan['duration']);card=future_card(c,forecast,dt)
        # 同一攻击沿用绝对事件时刻；若已过去，只标重新评估，不静默把事件推迟。
        event=prior['event_absolute_seconds'] if same else step*dt+card['rows'][0]['forecast_min_time_seconds']
        self.previous=dict(scene_id=str(scene),episode_id=episode,source_step=int(step),target_id=int(plan['attack_target_id']),
            strategy=plan['strategy'],anchors=copy.deepcopy(plan['anchors']),duration_frames=int(plan['duration']),
            event_absolute_seconds=event,accepted_not_execution_confirmed=True)
        return dict(updated=True,episode_id=episode,same_target_continuation=same,
            reason='continue_previous_target' if same else 'target_changed' if prior else previous_status.get('reason'),
            event_absolute_seconds=event,event_already_passed=event<=step*dt,execution_certified=False)


def bounded_future_proposals(ego,actor,target,forecast,duration=10,dt=.1,previous=None):
    """可解释规则只给4个稀疏锚点；候选必须继续经过既有背景/动力学/语义筛查。"""
    actor=np.asarray(actor,float);times=np.arange(4.)[:,None];result=[];seen=set()
    critical=[duration*dt/2,duration*dt,3.]
    if previous:
        remaining=previous['event_absolute_seconds']-forecast['source_step']*dt
        if .1<=remaining<=3:critical.append(float(remaining))
    for h in forecast['hypotheses']:
        for time in critical:
            point=forecast_point(forecast,h,time)
            raw=2*(point-actor[:2]-time*actor[2:4])/(time*time)
            for cap in (1.,2.,3.8):
                a=raw*min(1.,cap/max(float(np.linalg.norm(raw)),1e-12))
                anchors=actor[:2]+times*actor[2:4]+.5*times*times*a
                key=tuple(np.round(anchors,9).flat)
                if key in seen:continue
                seen.add(key);result.append(dict(target_id=int(target),strategy='lane_change',attack_family='merge_conflict',
                    anchors=anchors.tolist(),duration=duration,recipe='rolling_future_sparse_intent_v1',
                    recipe_basis=f'future_{h["id"]}_at_{time:.3f}s',ego_hypothesis=h['id'],event_relative_seconds=time))
    return result


def remaining_intent_candidate(previous,fresh,actor,step,dt=.1):
    """t0使用实测位置，仍覆盖的旧锚点平移消除跟踪误差；超出旧时域用新候选，明确标记。"""
    if fresh['target_id']!=previous['target_id']:return None
    elapsed=(step-previous['source_step'])*dt
    if not 0<elapsed<2:return None  # 至少有1秒旧预测可真正延续，不凭空延长旧计划。
    anchors=np.asarray(fresh['anchors'],float).copy();actor=np.asarray(actor,float)
    offset=actor[:2]-sample(previous['anchors'],[elapsed])[0];reused=[]
    for i in range(1,4):
        if elapsed+i<=3:
            anchors[i]=sample(previous['anchors'],[elapsed+i])[0]+offset;reused.append(i)
    anchors[0]=actor[:2]
    if not reused:return None
    result=copy.deepcopy(fresh);result.update(anchors=anchors.tolist(),recipe='remaining_world_intent_plus_fresh_tail_v1',
        prior_episode_id=previous['episode_id'],reused_relative_anchor_seconds=reused,
        fresh_tail_relative_anchor_seconds=[i for i in range(1,4) if i not in reused],
        tracking_offset_m=offset.tolist(),old_forecast_extrapolated=False)
    return result
