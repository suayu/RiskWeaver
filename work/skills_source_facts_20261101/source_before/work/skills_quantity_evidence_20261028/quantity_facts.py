"""只提供当前运动、候选比较与时间事实；不筛掉候选、不修改控制或原门控。"""
import math
import re
from numbers import Real


def finite(value):return isinstance(value,Real) and not isinstance(value,bool) and math.isfinite(value)


def comparison_groups(values,descending,tolerance=1e-6):
    """显式量名之外只作浮点排序；每组以首值比较，避免链式近似相等漂移。"""
    items=sorted(values.items(),key=lambda item:(-item[1] if descending else item[1],item[0]))
    groups=[];anchor=None
    for cid,value in items:
        if anchor is None or abs(value-anchor)>tolerance:
            groups.append([]);anchor=value
        groups[-1].append(cid)
    return groups


def motion_fact(ego,actor,target_id,step):
    if len(ego)<5 or len(actor)<5 or not all(finite(x) for x in list(ego[:5])+list(actor[:5])):
        raise ValueError('nonfinite_or_missing_motion_state')
    ego=list(map(float,ego[:5]));actor=list(map(float,actor[:5]))
    dx,dy=actor[0]-ego[0],actor[1]-ego[1];distance=math.hypot(dx,dy)
    yaw=ego[4];right=dx*math.sin(yaw)-dy*math.cos(yaw);forward=dx*math.cos(yaw)+dy*math.sin(yaw)
    rate=(dx*(actor[2]-ego[2])+dy*(actor[3]-ego[3]))/distance if distance>0 else None
    direction=('increasing_separating' if rate>1e-6 else 'decreasing_closing' if rate< -1e-6 else 'numerically_zero') if rate is not None else 'undefined_same_center'
    return dict(fact_id=f'actor{target_id}@step{step}:current_motion',target_id=target_id,source_step=step,
        right_m=right,forward_m=forward,heading_difference_degrees=abs(math.degrees(math.atan2(math.sin(actor[4]-yaw),math.cos(actor[4]-yaw)))),
        radial_gap_rate_mps=rate,current_distance_direction=direction,
        radial_numeric_tolerance_mps=1e-6,basis='source_positions_and_velocities_only',
        future_candidate_motion_not_current_motion=True,not_lane_topology=True)


def build_quantity_facts(state,context,step):
    """缺失或陈旧输入只标不可用，仍交给原流程，不引入新拒绝路径。"""
    if isinstance(step,bool) or not isinstance(step,int) or step<0:return dict(available=False,reason='invalid_source_step')
    if context.get('source_step')!=step or ('current_step' in state and state['current_step']!=step):
        return dict(available=False,reason='source_time_mismatch')
    candidates=context.get('candidates') or [];cards=context.get('incremental_cards') or {}
    actors={};issues=[];motions={};times={};increments={};margins={};missing_increment=[];missing_margin=[];ids=set()
    for row in state.get('agents',[]):
        ident=row.get('id')
        if isinstance(ident,bool) or not isinstance(ident,int) or ident in actors:
            return dict(available=False,reason='ambiguous_actor_identity')
        actors[ident]=row.get('state')
    for c in candidates:
        cid=c.get('candidate_id');target=c.get('target_id')
        if not isinstance(cid,str) or cid in ids:return dict(available=False,reason='ambiguous_candidate_identity')
        ids.add(cid)
        if isinstance(target,bool) or not isinstance(target,int) or target not in actors:
            issues.append('missing_target_state:'+cid)
        elif str(target) not in motions:
            try:motions[str(target)]=motion_fact(state['ego_state'],actors[target],target,step)
            except (ValueError,TypeError,KeyError):issues.append('invalid_motion_state:'+str(target))
        card=cards.get(cid) or {};value=card.get('min_gap_difference_m')
        if card.get('full_active_window_covered') is True and finite(value):increments[cid]=float(value)
        else:missing_increment.append(cid)
        margin=c.get('background_safety_margin_m')
        if finite(margin):margins[cid]=float(margin)
        else:missing_margin.append(cid)
        recipe_basis=c.get('recipe_basis');match=re.fullmatch(r'active_time_([0-9]+(?:\.[0-9]+)?)s',recipe_basis) if isinstance(recipe_basis,str) else None
        times[cid]=dict(active_horizon_seconds=float(card['active_seconds']) if finite(card.get('active_seconds')) else None,
            recipe_design_time_seconds=float(match[1]) if match else None,
            anchor_forecast_horizon_seconds=float(card['anchor_horizon_seconds']) if finite(card.get('anchor_horizon_seconds')) else None,
            recipe_design_time_is_not_attack_validity=True)
    return dict(available=True,source_step=step,actor_current_motion=motions,
        comparisons=dict(scope='all_current_published_candidates; not_a_strategy_recommendation',
            active_center_gap_CV_increment_m=dict(values=increments,larger_first_tie_groups=comparison_groups(increments,True),missing_or_incomplete=missing_increment),
            supplied_background_margin_m=dict(values=margins,smaller_first_tie_groups=comparison_groups(margins,False),missing_or_nonnumeric=missing_margin,
                missing_does_not_mean_zero_or_safe=True),numeric_tie_tolerance_m=1e-6),
        per_candidate_time=times,issues=issues,additional_acceptance_gate=False,candidate_order_changed=False,
        actual_risk_or_threat_certified=False,background_safety_certified=False)
