"""区分自然接近与意图增量的离线事实卡；不把运动代理称为真实攻击收益。"""
import math
import numpy as np


def _actor(state, target_id):
    actors = state['agents']
    ids = [a['id'] for a in actors]
    if len(ids) != len(set(ids)) or isinstance(target_id, bool) or target_id not in ids:
        raise ValueError('ambiguous_or_missing_background_actor')
    actor = np.asarray(next(a['state'] for a in actors if a['id'] == target_id), float)
    ego = np.asarray(state['ego_state'], float)
    if min(len(actor), len(ego)) < 4 or not np.isfinite(actor[:4]).all() or not np.isfinite(ego[:4]).all():
        raise ValueError('invalid_motion_evidence')
    return ego, actor


def incremental_card(state, candidate, dt=.1, anchor_interval=1.):
    """全程同一时刻比较候选/恒速目标与恒速ego；差值不是闭环因果效应。"""
    if not all(math.isfinite(v) and v > 0 for v in [dt, anchor_interval]):
        raise ValueError('invalid_evidence_timebase')
    ego, actor = _actor(state, candidate['target_id'])
    anchors = np.asarray(candidate['anchors'], float)
    frames = candidate['duration']
    if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or frames <= 0:
        raise ValueError('invalid_intent_duration')
    if anchors.ndim != 2 or anchors.shape[1] != 2 or len(anchors) < 2 or not np.isfinite(anchors).all():
        raise ValueError('invalid_sparse_anchors')
    horizon = (len(anchors)-1)*anchor_interval
    active = frames*dt
    covered = min(active, horizon)
    if np.linalg.norm(anchors[0]-actor[:2]) > .01:
        raise ValueError('source_anchor_does_not_match_target')
    times = np.linspace(0., covered, max(1, int(math.ceil(covered/dt)))+1)
    anchor_times = np.arange(len(anchors))*anchor_interval
    proposed = np.stack([np.interp(times, anchor_times, anchors[:, k]) for k in [0, 1]], axis=1)
    ego_xy = ego[:2]+times[:, None]*ego[2:4]
    natural = actor[:2]+times[:, None]*actor[2:4]
    proposal_gap = np.linalg.norm(proposed-ego_xy, axis=1)
    natural_gap = np.linalg.norm(natural-ego_xy, axis=1)
    pi, ni = int(np.argmin(proposal_gap)), int(np.argmin(natural_gap))
    increment = float(natural_gap[ni]-proposal_gap[pi])
    relative = actor[:2]-ego[:2]
    radial = float(relative@(actor[2:4]-ego[2:4])/max(np.linalg.norm(relative), 1e-12))
    # 保留两种时间口径，防止把不同时间的两个最小值误写为同一时刻的干预效果。
    return dict(candidate_id=candidate['candidate_id'], target_id=int(candidate['target_id']),
        target_role='background_vehicle', active_seconds=float(active),
        observed_proxy_seconds=float(covered), anchor_horizon_seconds=float(horizon),
        full_active_window_covered=bool(active <= horizon),
        source_center_gap_m=float(np.linalg.norm(relative)), radial_gap_rate_mps=radial,
        candidate_min_gap_m=float(proposal_gap[pi]), candidate_min_time_s=float(times[pi]),
        natural_CV_min_gap_m=float(natural_gap[ni]), natural_CV_min_time_s=float(times[ni]),
        min_gap_difference_m=increment,
        same_time_difference_at_candidate_min_m=float(natural_gap[pi]-proposal_gap[pi]),
        candidate_end_gap_m=float(proposal_gap[-1]), natural_CV_end_gap_m=float(natural_gap[-1]),
        proxy_direction=('closer_than_natural_CV' if increment > 1e-6 else
                         'farther_than_natural_CV' if increment < -1e-6 else 'no_min_gap_change'),
        basis='sparse_anchor_interpolation_vs_target_and_ego_constant_velocity',
        executed=False, actual_attack_benefit_measured=False, background_safety_certified=False,
        limitations=['center_gap_not_OBB_TTC_or_reachable_area', 'CV_not_real_policy_counterfactual',
                     'sparse_intent_not_executable_diffusion_trajectory', 'not_a_new_acceptance_gate'])


def factual_summary(card):
    """直接从事实卡渲染展示文本，原始LLM理由另存，不拿展示摘要掩盖其错误。"""
    return (f"候选{card['candidate_id']}，背景车{card['target_id']}；"
            f"意图有效{card['active_seconds']:.2f}秒，当前仅计算前{card['observed_proxy_seconds']:.2f}秒。"
            f"候选中心最近距{card['candidate_min_gap_m']:.3f}米（{card['candidate_min_time_s']:.2f}秒），"
            f"恒速参考最近距{card['natural_CV_min_gap_m']:.3f}米（{card['natural_CV_min_time_s']:.2f}秒）。"
            f"两最小值之差（参考减候选）{card['min_gap_difference_m']:+.3f}米；"
            f"候选最近时刻的同刻差值{card['same_time_difference_at_candidate_min_m']:+.3f}米。"
            "以上为运动代理，未测量实际攻击收益，未证明背景安全。")
