"""为同向后车候选补充活动期内的纵向闭合证据。"""
import math

import numpy as np


def closure_evidence(ego, actor, candidate, dt, anchor_interval):
    """使用源状态和稀疏意图，只描述已声明的活动窗。"""
    ego, actor = np.asarray(ego, float), np.asarray(actor, float)
    if min(ego.size, actor.size) < 7 or not np.isfinite([ego[:7], actor[:7]]).all():
        return None
    if np.any(ego[5:7] <= 0) or np.any(actor[5:7] <= 0):
        return None
    axis = np.array([math.cos(ego[4]), math.sin(ego[4])])
    side = np.array([-axis[1], axis[0]])
    delta = actor[:2] - ego[:2]
    angle = float(math.remainder(actor[4] - ego[4], 2 * math.pi))
    if delta @ axis >= 0 or abs(angle) > .7:
        return None
    lon_radius = .5 * (ego[5] + actor[5]*abs(math.cos(angle)) + actor[6]*abs(math.sin(angle)))
    lat_radius = .5 * (ego[6] + actor[5]*abs(math.sin(angle)) + actor[6]*abs(math.cos(angle)))
    if abs(delta @ side) > lat_radius:
        return None
    anchors = np.asarray(candidate["anchors"], float)
    duration = candidate["duration"]
    times = candidate.get("anchor_times_s")
    if times is None:
        times = np.arange(len(anchors)) * anchor_interval
    times = np.asarray(times, float)
    end = duration * dt
    if (len(anchors) < 2 or anchors.shape != (len(times), 2) or not np.isfinite(anchors).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or end > times[-1] + 1e-8 or times[0] > 0 or end <= 0):
        return None
    # 折线纵向净距在端点和折点取极值，且不把 1 秒之后的锚点算进效果。
    sample = np.unique(np.r_[0., times[(times > 0.) & (times < end)], end])
    target_xy = np.column_stack([np.interp(sample, times, anchors[:, i]) for i in (0,1)])
    reference = ego[:2] + sample[:, None] * ego[2:4]
    gap = (reference - target_xy) @ axis - lon_radius
    g0 = float(-delta @ axis - lon_radius)
    closing = float((actor[2:4]-ego[2:4]) @ axis)
    # 3.8 与原活动窗配方的加速度上限一致。这是恒速自车假设下的参考包络。
    envelope_end = g0 - closing*end - .5*3.8*end**2
    return dict(active_s=round(end, 3), initial_gap_m=round(g0, 3),
                initial_closing_mps=round(closing, 3), min_intent_longitudinal_gap_m=round(float(gap.min()), 3),
                end_intent_longitudinal_gap_m=round(float(gap[-1]), 3),
                best_case_end_gap_m_under_acceleration_cap=round(envelope_end, 3),
                envelope_can_close_in_window=bool(g0 <= 0 or envelope_end <= 0))


def add_context_evidence(context, env, dt, anchor_interval):
    """只增强模型事实，不改候选、有效期、回退和安全验证。"""
    candidates = context.get("candidates") or []
    actors = env.data_dict["agent"][-1]
    entries = {}
    for candidate in candidates:
        target = candidate["target_id"]
        if not (0 <= target < len(actors) and env.agent_active[target]):
            continue
        evidence = closure_evidence(env.ego_state, actors[target], candidate, dt, anchor_interval)
        if evidence is not None:
            entries[candidate["candidate_id"]] = evidence
    if entries:
        context["rear_window_evidence"] = entries
    return context


def install(planner_module):
    """进程内装配，实验结束或异常时恢复原调用路径。"""
    cls = planner_module.IncrementalSkillsPlanner
    original_context, original_render = cls.build_context, planner_module.render_incremental_prompt

    def context(self, env, state, planner, profile_context=None):
        built = original_context(self, env, state, planner, profile_context)
        return add_context_evidence(built, env, self.dt, self.interval)

    def render(prompt):
        text = original_render(prompt)
        if '"rear_window_evidence"' not in text:
            return text
        return ("For a rear-end intent, use rear_window_evidence for active-time longitudinal closure. "
                "Do not treat a smaller gap after expiry or a positive but insufficient closing speed as contact. "
                "An envelope or an anchor gap is a one-dimensional proxy, not an OBB contact or safety certificate. "
                "Prefer a feasible in-window interaction when available; preserve existing eligibility and safety rules.\n" + text)

    cls.build_context = context
    planner_module.render_incremental_prompt = render

    def restore():
        cls.build_context = original_context
        planner_module.render_incremental_prompt = original_render

    return restore

