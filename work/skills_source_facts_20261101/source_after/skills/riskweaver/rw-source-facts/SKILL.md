---
name: rw-source-facts
description: 使用当前测量方位与明确的攻击时钟解释场景，区分过去意图、当前事实和未来假设。
---

# 当前事实与时间解释

- `context.source_facts_v1` 是当前源帧确定性计算的事实。只有 `available=true` 时才引用；缺失不等于零、不存在、已经安全或不能攻击。
- 使用 `current_actor_positions` 的 `longitudinal_label` 与 `lateral_label` 描述当前方位。0.5米带内是 aligned，不因微小正负数说成左侧或右侧。当前径向速度描述靠近/远离；未来锚点趋近不代表车辆现在正在靠近。它们不是道路或车道拓扑标签。
- `prior_intent_clock.age_since_start_seconds` 是距上次意图开始的时间，绝不是过期时长。只有 `nominal_lease_expired=true` 时，才可用 `age_since_expiration_seconds` 描述“有效期结束多久”；到期边界的过期时长为零。有效期结束不等于已观测到执行停止。
- `remaining_lease_seconds` 是当前剩余有效期。`previous_event_relative_to_now_seconds` 是先前假设事件距当前的时间，可为负，不是本次指令的有效期，也不是自车的真实未来。
- 历史意图是已接收的计划，不是已执行证据。没有可用历史不等于从未攻击。发生过冲撞、已执行到某锚点或造成风险变化，必须另有实际执行记录支持。
- 候选数值比较必须明确量名和比较范围，使用 supplied comparison facts。最小间距、同一时刻间距差、背景安全裕度、攻击有效期和预测时域互不替代。
- 仍按原响应 schema 输出 scene_analysis 与 candidate_reasons；不要增加字段、改写身份或编造候选。规则帮助推理，不证明战术最优、持续可执行或必然威胁自车。
