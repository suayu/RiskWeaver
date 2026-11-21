"""完整锚点时域与可切换的动力学插值执行。"""

import math

import numpy as np


ANCHOR_TIMES = (0., 1., 2., 3.)
MAX_CURVATURE = .2
MAX_LATERAL_ACCELERATION = 2.5


def window_steps(dt):
    """实际仿真步必须准确落在全部秒级锚点上。"""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid_anchor_dt")
    steps = round(1. / dt)
    if steps < 1 or not math.isclose(steps * dt, 1., abs_tol=1e-8):
        raise ValueError("anchor_times_off_simulation_grid")
    return 3 * steps


def interpolation_reference(anchors, initial_velocity, times):
    """分段三次 Hermite 插值。首点接当前速度，内部斜率取相邻割线均值。"""
    anchors = np.asarray(anchors, dtype=float)
    initial_velocity = np.asarray(initial_velocity, dtype=float)
    times = np.asarray(times, dtype=float)
    if (anchors.shape != (4, 2) or initial_velocity.shape != (2,)
            or not np.isfinite(anchors).all() or not np.isfinite(initial_velocity).all()
            or not np.isfinite(times).all() or (times < 0).any()):
        raise ValueError("invalid_anchor_interpolation_input")
    slopes = np.diff(anchors, axis=0)
    velocities = np.vstack((initial_velocity, (slopes[:-1] + slopes[1:]) / 2, slopes[-1]))
    segment = np.minimum(np.floor(times).astype(int), 2)
    u = (np.minimum(times, 3.) - segment)[:, None]
    p0, p1 = anchors[segment], anchors[segment + 1]
    v0, v1 = velocities[segment], velocities[segment + 1]
    result = ((2*u**3 - 3*u**2 + 1)*p0 + (u**3 - 2*u**2 + u)*v0
              + (-2*u**3 + 3*u**2)*p1 + (u**3 - u**2)*v1)
    # 3 秒后的恒速尾段只用于联合预测与评估，不继续锚点攻击。
    return result + np.maximum(times - 3., 0.)[:, None] * velocities[-1]


def check_dynamics(positions, state, dt, limits, previous_velocity=None):
    """按仿真器的速度差分口径检查整段，也检查重规划边界的 jerk。"""
    velocity = np.diff(np.vstack((state[:2], positions)), axis=0) / dt
    history = [state[2:4]] if previous_velocity is None else [previous_velocity, state[2:4]]
    acceleration = np.diff(np.vstack((*history, velocity)), axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    peak = lambda values: float(np.linalg.norm(values, axis=1).max(initial=0.))
    speed = peak(velocity)
    peaks = dict(max_speed=speed, max_step_distance=speed * dt,
                 max_acceleration=peak(acceleration[-len(velocity):]), max_jerk=peak(jerk))
    valid = all(math.isfinite(value) and value <= limits[name] + 1e-9
                for name, value in peaks.items())
    return velocity, peaks, valid


def forward_headings(velocity, state, dt):
    """禁止倒车和原地旋转，检查由位移方向导出的曲率与侧向加速度。"""
    heading = float(state[4])
    speed = float(np.linalg.norm(state[2:4]))
    result = []
    for vx, vy in velocity:
        next_speed = math.hypot(vx, vy)
        if next_speed >= 1e-6:
            direction = math.atan2(vy, vx)
            delta = math.remainder(direction - heading, 2 * math.pi)
            mean_speed = (speed + next_speed) * .5
            if (abs(delta) > MAX_CURVATURE * mean_speed * dt + 1e-8
                    or next_speed * abs(delta) / dt > MAX_LATERAL_ACCELERATION + 1e-8):
                return None
            heading += delta
        result.append(heading)
        speed = next_speed
    return np.asarray(result)


def pursuit_rollout(reference, state, dt, limits, previous_velocity, hold_speed):
    """以转向量优先的有界搜索逐步追踪，不向车身后方平移。"""
    point, velocity = state[:2].copy(), state[2:4].copy()
    heading = float(state[4])
    accel = None if previous_velocity is None else (velocity - previous_velocity) / dt
    speed_limit = min(limits["max_speed"], limits["max_step_distance"] / dt)
    points = []
    lookahead = max(1, round(.5 / dt))
    for index in range(len(reference)):
        delta = reference[min(index + lookahead - 1, len(reference) - 1)] - point
        speed = float(np.linalg.norm(velocity))
        distance = float(np.linalg.norm(delta))
        direction = math.atan2(delta[1], delta[0]) if distance > 1e-6 else heading
        error = math.remainder(direction - heading, 2 * math.pi)
        if abs(abs(error) - math.pi) < 1e-6:
            error = math.pi  # 锚点正后方时固定向左打方向。
        lower = max(0., speed - limits["max_acceleration"] * dt)
        upper = min(speed_limit, speed + limits["max_acceleration"] * dt)
        if upper < lower:
            return None
        desired = min(speed_limit, distance / (lookahead * dt))
        if abs(error) > .3:
            desired = min(max(2., desired), math.sqrt(MAX_LATERAL_ACCELERATION / MAX_CURVATURE))
        speeds = np.unique(np.r_[np.linspace(lower, upper, 17), np.clip(desired, lower, upper),
                                 np.clip(speed, lower, upper)])
        if hold_speed and speed >= 1.:
            if speed > speed_limit:
                return None
            speeds = np.array([speed])
        next_speeds = speeds[:, None]
        yaw_limit = np.minimum(MAX_CURVATURE * (speed + next_speeds) * .5 * dt,
                               MAX_LATERAL_ACCELERATION * dt / np.maximum(next_speeds, 1e-6))
        # 允许为满足 jerk 暂时沿上一时刻的转向继续，避免跨过目标方位时突兀反打。
        angles = np.concatenate((yaw_limit * np.linspace(-1., 1., 33),
                                 np.clip(error, -yaw_limit, yaw_limit)), axis=1)
        headings = heading + angles
        trials = next_speeds[..., None] * np.stack((np.cos(headings), np.sin(headings)), axis=-1)
        trial_accel = (trials - velocity) / dt
        valid = np.linalg.norm(trial_accel, axis=-1) <= limits["max_acceleration"] - 1e-8
        if accel is not None:
            valid &= np.linalg.norm((trial_accel - accel) / dt, axis=-1) <= limits["max_jerk"] - 1e-8
        rows, columns = np.nonzero(valid)
        if not len(rows):
            return None
        # 优先选朝向误差减少最多的动作，再接近参考速度，最后减少加速度。
        key = np.lexsort((np.linalg.norm(trial_accel[rows, columns], axis=-1),
                          np.abs(speeds[rows] - desired), np.abs(error - angles[rows, columns])))
        i, j = rows[key[0]], columns[key[0]]
        velocity, accel = trials[i, j], trial_accel[i, j]
        if speeds[i] >= 1e-6:
            heading = float(headings[i, j])
        point = point + velocity * dt
        points.append(point.copy())
    return np.asarray(points)


def pursue_forward(reference, state, dt, limits, previous_velocity=None):
    """局部最强转向无法延续整段时，再检查保持速度的有界转向轨迹。"""
    for hold_speed in (False, True):
        path = pursuit_rollout(reference, state, dt, limits, previous_velocity, hold_speed)
        if path is not None:
            return path
    return None


def interpolate_joint(env):
    """目标车插值，其余车以当前恒速形成联合预测，供原执行链复核。"""
    from policies.ego_profile import JointTrajectorySafetyGate
    from policies.joint_safety import NoSafeJointCandidate, validate_execution_prefix
    from policies.traffic_types import JointTrajectory

    cfg = env.cfg.sim.traffic_model
    states = np.asarray(env.data_dict["agent"][-1], dtype=float)
    ids = np.flatnonzero(env.agent_active)
    current = states[ids]
    dt = float(env.dt)
    horizon = int(cfg.prediction_horizon)
    offsets = np.arange(1, horizon + 1) * dt
    velocity = np.repeat(current[:, None, 2:4], horizon, axis=1)
    positions = current[:, None, :2] + velocity * offsets[None, :, None]
    yaw = np.repeat(current[:, None, 4:5], horizon, axis=1)
    intent = env.attack_intent
    meta = dict(backend="kinematic_interpolation", background_prediction="constant_velocity",
                anchor_guidance_active=False)
    if intent is not None and intent["target_id"] not in ids:
        raise NoSafeJointCandidate("interpolation_target_inactive")
    if intent is not None:
        target = int(intent["target_id"])
        row = int(np.searchsorted(ids, target))
        elapsed = int(env.current_step) - int(intent["source_step"])
        if not 0 <= elapsed < window_steps(dt) or intent["duration"] != window_steps(dt):
            raise NoSafeJointCandidate("interpolation_invalid_execution_window")
        if elapsed == 0:
            if not np.allclose(current[row, :2], intent["anchors"][0], rtol=0., atol=.01):
                raise NoSafeJointCandidate("interpolation_anchor_source_mismatch")
            intent["interpolation_initial_velocity"] = current[row, 2:4].tolist()
        reference = interpolation_reference(intent["anchors"], intent["interpolation_initial_velocity"],
                                             elapsed * dt + offsets)
        limits = JointTrajectorySafetyGate(env.cfg.sim).dynamic_limits
        settings = cfg.dynamics_projection
        previous = None
        # 起点之前的日志速度未经过同一仿真器差分；仅拼接本计划已执行的速度。
        if elapsed:
            previous = np.asarray(env.data_dict["agent"][-2], dtype=float)[target, 2:4]
        v, peaks, valid = check_dynamics(reference, current[row], dt, limits, previous)
        result = reference
        headings = forward_headings(v, current[row], dt)
        fallback_reason = None if headings is not None else "interpolation_nonholonomic_reference"
        projected = False
        if not valid and fallback_reason is None:
            from policies.trajectory_projection import project_positions

            try:
                result, _, _, _ = project_positions(
                    reference, current[row], dt, limits, float(settings.max_seconds_per_vehicle),
                    float(settings.max_deviation_m))
                projected = True
                v, peaks, valid = check_dynamics(result, current[row], dt, limits, previous)
                headings = forward_headings(v, current[row], dt)
                if not valid or headings is None:
                    fallback_reason = "interpolation_projection_dynamics_postcheck_failed"
            except NoSafeJointCandidate as error:
                fallback_reason = str(error)
        if fallback_reason is not None:
            result = pursue_forward(reference, current[row], dt, limits, previous)
            if result is None:
                raise NoSafeJointCandidate("interpolation_forward_pursuit_infeasible")
            v, peaks, valid = check_dynamics(result, current[row], dt, limits, previous)
            headings = forward_headings(v, current[row], dt)
            if not valid or headings is None:
                raise NoSafeJointCandidate("interpolation_forward_pursuit_postcheck_failed")
        positions[row], velocity[row], yaw[row, :, 0] = result, v, headings
        anchor_errors = {}
        for t in ANCHOR_TIMES[1:]:
            k = round(t / dt) - elapsed - 1
            if 0 <= k < horizon:
                anchor_errors[str(t)] = float(np.linalg.norm(result[k] - intent["anchors"][int(t)]))
        deviation = float(np.linalg.norm(result - reference, axis=1).max())
        within_anchor_budget = (deviation <= float(settings.max_deviation_m) and all(
            error <= float(settings.max_deviation_m) for error in anchor_errors.values()))
        if fallback_reason is None and not within_anchor_budget:
            raise NoSafeJointCandidate("interpolation_anchor_deviation")
        meta.update(anchor_guidance_active=True, anchor_guidance_target_id=target,
                    planner_request_id=intent.get("request_id"), planner_strategy=intent.get("strategy"),
                    anchor_guidance_valid_steps=window_steps(dt) - elapsed, anchor_times_s=list(ANCHOR_TIMES),
                    interpolation=dict(source_step=int(intent["source_step"]), projected=projected,
                                       follow_mode="forward_pursuit" if fallback_reason else "interpolation",
                                       fallback_reason=fallback_reason, within_anchor_budget=within_anchor_budget,
                                       peaks=peaks, anchor_errors_m=anchor_errors, max_deviation_m=deviation))
    joint = JointTrajectory(env.current_step, ids, positions, yaw, velocity,
                            np.ones((len(ids), horizon), dtype=bool), meta)
    if bool(getattr(env.cfg.sim, 'joint_trajectory_gate_enabled', True)):
        validate_execution_prefix(states, joint, env.get_scenario_frame().static_obstacles_global)
    env.inject_joint_trajectory(joint)
    return joint
