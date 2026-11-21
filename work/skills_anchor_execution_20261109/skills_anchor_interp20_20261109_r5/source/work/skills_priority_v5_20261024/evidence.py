"""给离散规划器提供可引用事实；所有距离只是锚点/CV代理，不是假想的执行结果。"""
import math
import numpy as np


def candidate_evidence(state, candidates, dt, interval):
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(interval) or interval <= 0:
        raise ValueError('invalid_evidence_timebase')
    ego = np.asarray(state['ego_state'], float)
    if ego.shape[0] < 4 or not np.isfinite(ego[:4]).all():
        raise ValueError('invalid_ego_evidence')
    actors = {int(x['id']): x for x in state['agents']}
    output = {}
    for c in candidates:
        cid = c['candidate_id']; target = c['target_id']
        if cid in output or isinstance(target, bool) or target not in actors:
            raise ValueError('invalid_or_duplicate_candidate_identity')
        anchors = np.asarray(c['anchors'], float)
        duration = c['duration']
        if anchors.ndim != 2 or anchors.shape[1] != 2 or len(anchors) < 2 or not np.isfinite(anchors).all():
            raise ValueError('invalid_anchor_evidence')
        if isinstance(duration, bool) or not isinstance(duration, (int, np.integer)) or duration <= 0:
            raise ValueError('invalid_duration_evidence')
        horizon = (len(anchors) - 1) * interval
        active = min(duration * dt, horizon)

        def distances(start, end):
            # 固定0.1秒/仿真dt采样并包含边界，不把空间最近点当作时间对齐。
            n = max(1, int(math.ceil((end-start) / dt)))
            times = np.linspace(start, end, n+1)
            anchors_t = np.arange(len(anchors)) * interval
            xy = np.stack([np.interp(times, anchors_t, anchors[:, axis]) for axis in (0, 1)], axis=1)
            ego_xy = ego[:2] + times[:, None] * ego[2:4]
            gap = np.linalg.norm(xy-ego_xy, axis=1)
            best = int(np.argmin(gap))
            return dict(start_seconds=float(start), end_seconds=float(end),
                min_center_distance_m=float(gap[best]), time_of_min_seconds=float(times[best]),
                start_center_distance_m=float(gap[0]), end_center_distance_m=float(gap[-1]),
                reduction_m=float(gap[0]-gap[-1]), basis='time_aligned_anchor_interpolation_vs_ego_CV',
                executed=False, collision_or_evasion_guaranteed=False)

        facts = {
            cid+':actor': dict(target_id=int(target), role='background_vehicle', ego_reference='separate_ego'),
            cid+':validity': dict(active_horizon_seconds=duration*dt, anchor_horizon_seconds=horizon,
                duration_frames=int(duration), frame_dt_seconds=dt,
                complete_active_window_covered_by_anchors=duration*dt <= horizon),
            cid+':active': distances(0., active),
            cid+':safety': dict(assessment=c.get('background_safety_assessment'),
                supplied_margin_m=c.get('background_safety_margin_m'),
                semantic_metrics=c.get('semantic_evidence'),
                final_joint_validation_required=True, positive_ego_area_certified=False),
        }
        if horizon > active:
            # 到期边界仍属于有效期，不可在“到期以后”重复计入同一个锚点。
            facts[cid+':after'] = distances(min(active+dt, horizon), horizon)
        output[cid] = dict(target_id=int(target), active_horizon_seconds=duration*dt, facts=facts)
    return output


def validate_grounding(response, catalog, evidence):
    """校验对象/时域和事实引用，不能把文本理由非空误称为战术最优或真实威胁。"""
    ranked = response['ranking']; grounding = response.get('grounding')
    if not isinstance(grounding, dict) or set(grounding) != set(ranked):
        raise ValueError('priority_grounding_keys')
    for cid in ranked:
        row = grounding[cid]; expected = evidence[cid]
        required = {'target_id', 'target_role', 'active_horizon_seconds', 'evidence_ids', 'interaction_basis'}
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError('priority_grounding_schema')
        if isinstance(row['target_id'], bool) or row['target_id'] != catalog[cid]['target_id'] or row['target_role'] != 'background_vehicle':
            raise ValueError('priority_actor_role_mismatch')
        value = row['active_horizon_seconds']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value-expected['active_horizon_seconds']) > 1e-8:
            raise ValueError('priority_active_horizon_mismatch')
        refs = row['evidence_ids']
        if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs) or len(refs) != len(set(refs)) or not set(refs) <= set(expected['facts']):
            raise ValueError('priority_invented_evidence')
        if not {cid+':actor', cid+':validity'} <= set(refs):
            raise ValueError('priority_missing_identity_or_time_evidence')
        basis = row['interaction_basis']
        if basis not in {'active_window_proxy', 'later_forecast_only', 'exploratory_probe'}:
            raise ValueError('priority_invalid_interaction_basis')
        required_ref = cid+':after' if basis == 'later_forecast_only' else cid+':active'
        if required_ref not in refs:
            raise ValueError('priority_missing_interaction_evidence')
