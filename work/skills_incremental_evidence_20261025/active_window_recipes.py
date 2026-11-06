"""仅供离线机会审计的稀疏意图配方；未接入当前冻结闭环，不输出连续控制。"""
import math
import numpy as np


def proposals_for_actor(ego, actor, target_id, dt=.1, duration=10, cap_values=(1., 2., 3.8)):
    """使用当前运动状态寻找有效期内交互，不读取未来状态或碰撞结果。"""
    ego, actor = np.asarray(ego, float), np.asarray(actor, float)
    if min(len(ego), len(actor)) < 5 or not np.isfinite(ego[:5]).all() or not np.isfinite(actor[:5]).all():
        raise ValueError('invalid_source_state')
    if isinstance(target_id, bool) or not isinstance(target_id, (int, np.integer)):
        raise ValueError('invalid_background_identity')
    if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 30:
        raise ValueError('recipe_requires_covered_active_duration')
    if not math.isfinite(dt) or dt <= 0 or duration*dt > 3.:
        raise ValueError('recipe_requires_covered_timebase')
    if not cap_values or any(not math.isfinite(c) or not 0 < c <= 3.8 for c in cap_values):
        raise ValueError('unchanged_sparse_acceleration_cap_required')
    speed = float(np.linalg.norm(actor[2:4]))
    heading = actor[2:4]/speed if speed > 1e-6 else np.array([math.cos(actor[4]), math.sin(actor[4])])
    angle = abs(math.atan2(math.sin(actor[4]-ego[4]), math.cos(actor[4]-ego[4])))
    motion_relation = 'opposing_headings' if angle > 2.4 else 'similar_headings' if angle < .7 else 'crossing_headings'
    t = np.arange(4.)[:, None]
    accelerations = []
    # 纵向推进补充既有扩展中只有制动和三秒末端对齐的偏置；不降低筛查阈值。
    for cap in cap_values:
        accelerations.append(('sudden_acceleration', 'longitudinal_pressure', heading*cap, 'advance_along_current_motion'))
    for critical_time in (duration*dt/2., duration*dt):
        point = ego[:2]+critical_time*ego[2:4]
        raw = 2*(point-actor[:2]-critical_time*actor[2:4])/(critical_time**2)
        for cap in cap_values:
            a = raw*min(1., cap/max(np.linalg.norm(raw), 1e-12))
            accelerations.append(('lane_change', 'active_window_conflict', a, f'active_time_{critical_time:.3f}s'))
    result, seen = [], set()
    for strategy, family, acceleration, source in accelerations:
        if np.linalg.norm(acceleration) < 1e-9:
            continue  # 不把完全等于恒速的参考轨迹伪装成新攻击。
        anchors = actor[:2]+t*actor[2:4]+.5*t*t*acceleration
        key = tuple(np.round(anchors, 9).reshape(-1))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(candidate_id=f'a{int(target_id)}_{len(result)}', target_id=int(target_id),
            strategy=strategy, attack_family=family, anchors=anchors.tolist(), duration=duration,
            recipe='active_window_sparse_intent_offline_v1', recipe_basis=source,
            motion_relation=motion_relation, relation_scope='kinematic_heading_only_not_map_topology',
            screened=False, execution_approved=False))
    return result


def screened_proposals(env, state, builder, validator):
    """原背景/动力学screen和原语义验证都通过才发布；拒绝原件全部保留。"""
    states = np.asarray(env.data_dict['agent'][-1], float)
    lanes = [np.asarray(line)[:, :2] for line in env.scenario_dict.get('lanes', []) if len(line) > 1]
    accepted, audit = [], []
    for actor in state['agents']:
        target = actor['id']
        if isinstance(target, bool) or not isinstance(target, (int, np.integer)) or not 0 <= target < len(states) or not env.agent_active[target]:
            audit.append(dict(target_id=target, accepted=False, reason='missing_or_inactive_source_actor'))
            continue
        for proposal in proposals_for_actor(env.ego_state, states[target], target, float(env.dt)):
            estimates, reason = builder.screen(env, target, np.asarray(proposal['anchors']), lanes)
            entry = dict(candidate=proposal, accepted=False, screen_rejection=reason)
            if reason is None:
                semantic = validator(state, proposal)
                entry['semantic_validation'] = semantic
                if semantic['accepted']:
                    accepted.append(dict(proposal, **estimates, semantic_evidence=semantic['metrics'], screened=True))
                    entry['accepted'] = True
            audit.append(entry)
    return accepted, audit
