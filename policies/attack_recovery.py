"""拒绝只结束当前动态尝试，保留场景和全部预算及失败证据。"""
import numpy as np
from policies.traffic_types import JointTrajectory


def cancel_attempt(env, generator, trace, reason):
    intent = env.attack_intent or {}
    event = dict(kind="attack_attempt_rejected", step=int(env.current_step), reason=reason,
                 request_id=intent.get("request_id"), target_id=intent.get("target_id"))
    trace.append(event)
    runtime = getattr(generator, "rag_runtime", None)
    if runtime is not None:
        runtime.note_failure(reason)
    if intent.get('target_id') is not None:
        env.rejected_dynamic_attempt = dict(scene_id=str(env.current_scene_id),
                                           source_step=int(env.current_step), reason=reason)
    env.clear_attack_intent()
    env.pending_joint_trajectory = None
    env.latest_diffusion_candidate_trajectories = []
    generator.llm_anchors = None
    generator.diffusion_trajectory = None
    generator.attack_duration = 0
    generator._reachability_query_pending = False
    return event


def passive_joint(env):
    """只从当前真实状态生成CV未来，不复用被拒绝的攻击候选。"""
    states = np.asarray(env.data_dict["agent"][-1], dtype=float)
    ids = np.flatnonzero(env.agent_active)
    current = states[ids]
    horizon = int(env.cfg.sim.traffic_model.prediction_horizon)
    times = np.arange(1, horizon + 1) * float(env.dt)
    velocity = np.repeat(current[:, None, 2:4], horizon, axis=1)
    positions = current[:, None, :2] + velocity * times[None, :, None]
    yaw = np.repeat(current[:, None, 4:5], horizon, axis=1)
    return JointTrajectory(env.current_step, ids, positions, yaw, velocity,
                           np.ones((len(ids), horizon), dtype=bool),
                           dict(backend="rejected_attack_passive_cv", anchor_guidance_active=False))
