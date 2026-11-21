"""零面积时搜索并保守验证规避轨迹。有限搜索失败不表示实际不可规避。"""

import itertools
import math
import time

import numpy as np


def inside_route_tube(points, route, radius):
    """每段中心连线都在同一凸胶囊内，圆形车体包络始终留在路线走廊。"""
    delta = np.diff(route, axis=0)
    length2 = np.sum(delta**2, axis=1)
    keep = length2 > 1e-12
    delta, length2, starts = delta[keep], length2[keep], route[:-1][keep]
    offset = points[:, None, :] - starts[None, :, :]
    u = np.clip(np.sum(offset * delta, axis=2) / length2, 0., 1.)
    inside = np.sum((offset - u[..., None] * delta)**2, axis=2) <= radius**2
    return bool(np.all(np.any(inside[:-1] & inside[1:], axis=1)))


def swept_clear(points, yaw, extent, traffic, traffic_yaw, traffic_extent):
    """中点 SAT 轴加整段相对位移和旋转上界，保守检查连续运动。"""
    if not len(traffic):
        return True
    ego_yaw = (yaw[1:] + yaw[:-1]) * .5
    other_yaw = (traffic_yaw[:, 1:] + traffic_yaw[:, :-1]) * .5
    heading = np.stack((np.broadcast_to(ego_yaw, other_yaw.shape), other_yaw), axis=-1)
    forward = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
    left = np.stack((-forward[..., 1], forward[..., 0]), axis=-1)
    axes = np.concatenate((forward, left), axis=-2)
    relative = traffic - points[None, :, :]
    middle = (relative[:, 1:] + relative[:, :-1]) * .5
    drift = np.diff(relative, axis=1) * .5
    projection = lambda x: np.abs(np.einsum("ntai,nti->nta", axes, x))
    length = np.column_stack((np.full(len(traffic), extent[0]), traffic_extent[:, 0]))
    width = np.column_stack((np.full(len(traffic), extent[1]), traffic_extent[:, 1]))
    radii = (np.abs(np.einsum("ntai,ntbi->ntab", axes, forward)) * length[:, None, None, :]
             + np.abs(np.einsum("ntai,ntbi->ntab", axes, left)) * width[:, None, None, :]).sum(axis=-1) * .5
    rotation = (.25 * np.linalg.norm(extent) * np.abs(np.diff(yaw))[None, :]
                + .25 * np.linalg.norm(traffic_extent, axis=1)[:, None] * np.abs(np.diff(traffic_yaw, axis=1)))
    return bool(np.all(np.any(projection(middle) - projection(drift) - radii > rotation[..., None] + 1e-7,
                              axis=-1)))


def simulate(ego, times, first, second, accel, brake, max_speed, initial_a, initial_curvature):
    """连续线段参考以实测状态起步，逐步限制 jerk、速度和侧向加速度。"""
    dt = float(times[1])
    point = ego[:2].copy()
    heading = float(ego[4])
    speed = float(np.linalg.norm(ego[2:4]))
    a, curvature = initial_a, initial_curvature
    lateral = speed**2 * curvature
    points, yaws = [point.copy()], [heading]
    for t in times[:-1]:
        want, steering = first if t < 1. else second
        # 用有限 jerk 接入初始零加速度，达到限速或停住时也检查实际速度差。
        requested = a + float(np.clip(want - a, -12. * dt, 12. * dt))
        next_speed = float(np.clip(speed + requested * dt, 0., max_speed))
        applied = (next_speed - speed) / dt
        longitudinal = applied / (accel if applied >= 0 else brake)
        lateral_cap = 2.5 * math.sqrt(max(0., 1. - longitudinal**2))
        limit = lateral_cap / max(speed, next_speed, .1)**2
        next_curvature = float(np.clip(steering, curvature - .35*dt, curvature + .35*dt))
        next_curvature = float(np.clip(next_curvature, -limit, limit))
        next_lateral = next_speed**2 * next_curvature
        if (abs(next_curvature - curvature) > .35 * dt + 1e-8
                or math.hypot(applied - a, next_lateral - lateral) > 12. * dt + 1e-8):
            return None
        a, curvature, lateral = applied, next_curvature, next_lateral
        distance = (speed + next_speed) * .5 * dt
        change = distance * curvature
        point = point + distance * np.array([math.cos(heading + change / 2), math.sin(heading + change / 2)])
        heading += change
        speed = next_speed
        points.append(point)
        yaws.append(heading)
    return np.asarray(points), np.asarray(yaws)


def forecast_geometry(env, times):
    """只使用当前源帧及其真实联合预测，不把缺测当作无障碍。"""
    joint = env.pending_joint_trajectory
    states = np.asarray(env.data_dict["agent"][-1], dtype=float)
    active = np.flatnonzero(env.agent_active)
    if joint is None or joint.source_step != env.current_step or not np.array_equal(joint.agent_ids, active):
        raise ValueError("missing_or_misaligned_joint")
    required = math.ceil(times[-1] / env.dt - 1e-8)
    if joint.positions_global.shape[1] < required or not joint.valid_mask[:, :required].all():
        raise ValueError("incomplete_joint_horizon")
    native = np.arange(required + 1) * env.dt
    positions, yaws, extents = [], [], []
    for row, actor in enumerate(active):
        path = np.vstack((states[actor, :2], joint.positions_global[row, :required]))
        yaw = np.unwrap(np.r_[states[actor, 4], joint.yaws_global[row, :required, 0]])
        positions.append(np.column_stack([np.interp(times, native, p) for p in path.T]))
        yaws.append(np.interp(times, native, yaw))
        extents.append(states[actor, 5:7])
    for obstacle in env.get_static_obstacles():
        positions.append(np.repeat([obstacle["center"]], len(times), axis=0))
        yaws.append(np.full(len(times), obstacle["yaw"]))
        extents.append([obstacle["length"], obstacle["width"]])
    geometry = (np.asarray(positions, float).reshape(-1, len(times), 2),
                np.asarray(yaws, float).reshape(-1, len(times)), np.asarray(extents, float).reshape(-1, 2))
    if any(not np.isfinite(value).all() for value in geometry) or (geometry[2] <= 0).any():
        raise ValueError("invalid_obstacle_geometry")
    return geometry


def search(env, max_accel, seconds=1.5):
    """对首秒与后续分别搜索控制，超时或未通过保守几何复核一律不放行。"""
    cfg = env.cfg.sim.evaluation.reachability
    horizon = float(cfg.horizon_seconds)
    if not (math.isfinite(horizon) and 1. <= horizon <= 4. and math.isfinite(max_accel) and 2. < max_accel <= 6.):
        raise ValueError("unsupported_witness_envelope")
    dt = .05
    if (not math.isclose(round(horizon / dt) * dt, horizon, abs_tol=1e-9)
            or not math.isclose(round(env.dt / dt) * dt, env.dt, abs_tol=1e-9)):
        return dict(verified=False, status="unsupported_witness_time_grid")
    times = np.arange(round(horizon / dt) + 1) * dt
    ego = np.asarray(env.ego_state, dtype=float)
    route = np.asarray(env.scenario_dict["route"], dtype=float)[:, :2]
    width = float(cfg.lane_half_width)
    # 车体外接圆与中心线走廊一起提供保守道路证书，不扩大原有横向范围。
    tube_radius = width - float(np.linalg.norm(ego[5:7])) / 2
    max_speed = float(cfg.max_speed)
    if (not np.isfinite(ego[:7]).all() or (ego[5:7] <= 0).any() or not np.isfinite(route).all()
            or len(route) < 2 or not math.isfinite(tube_radius) or tube_radius <= 0
            or not math.isfinite(max_speed) or not 0 < max_speed
            or np.linalg.norm(ego[2:4]) > max_speed):
        return dict(verified=False, status="invalid_witness_geometry")
    try:
        traffic = forecast_geometry(env, times)
    except (ValueError, TypeError, KeyError) as error:
        return dict(verified=False, status="prediction_unavailable", detail=str(error))
    brake = -min(float(x) for x in cfg.longitudinal_accelerations)
    if not math.isfinite(brake) or brake <= 0:
        return dict(verified=False, status="invalid_witness_brake_limit")
    initial_a, initial_curvature = 0., 0.
    speed = float(np.linalg.norm(ego[2:4]))
    heading = np.array([math.cos(ego[4]), math.sin(ego[4])])
    if np.linalg.norm(ego[2:4] - speed*heading) > .1:
        return dict(verified=False, status="witness_initial_velocity_heading_mismatch")
    if env.current_step > 0:
        history = env.data_dict.get("ego", [])
        if len(history) < 2:
            return dict(verified=False, status="missing_ego_motion_history")
        previous = np.asarray(history[-2], dtype=float).reshape(-1)
        if previous.size < 5 or not np.isfinite(previous[:5]).all():
            return dict(verified=False, status="invalid_ego_motion_history")
        prior_speed = float(np.linalg.norm(previous[2:4]))
        initial_a = (speed - prior_speed) / env.dt
        delta_yaw = math.remainder(float(ego[4] - previous[4]), 2 * math.pi)
        initial_curvature = delta_yaw / max(.1, (speed + prior_speed) * .5) / env.dt
        if not (-brake <= initial_a <= max_accel and abs(initial_curvature) <= .12
                and (initial_a / (max_accel if initial_a >= 0 else brake))**2
                + (speed**2 * initial_curvature / 2.5)**2 <= 1.):
            return dict(verified=False, status="initial_motion_outside_witness_envelope")
    longitudinal = tuple(dict.fromkeys((0., max_accel, .75*max_accel, .5*max_accel, -brake, -.5*brake)))
    steering = (0., -.06, .06, -.12, .12)
    first = list(itertools.product(steering, longitudinal))
    # 先尝试固定控制，然后查找加速后保持、加速后刹车及变向。
    controls = [(a, a) for a in first] + [(a, b) for a in first
                                        for b in [(0., 0.), (0., max_accel), (0., -brake),
                                                  (-.06, 0.), (.06, 0.)] if a != b]
    started = time.monotonic()
    tested = 0
    for (k1, a1), (k2, a2) in controls:
        if time.monotonic() - started >= seconds:
            return dict(verified=False, status="search_timeout", tested=tested)
        tested += 1
        result = simulate(ego, times, (a1, k1), (a2, k2), max_accel, brake, max_speed,
                          initial_a, initial_curvature)
        if result is None:
            continue
        points, yaw = result
        if inside_route_tube(points, route, tube_radius) and swept_clear(points, yaw, ego[5:7], *traffic):
            if time.monotonic() - started >= seconds:
                return dict(verified=False, status="search_timeout", tested=tested)
            return dict(verified=True, status="grid_empty_witness_feasible", tested=tested,
                        source_step=int(env.current_step), step_seconds=float(times[1]), horizon_seconds=horizon,
                        max_acceleration_mps2=max_accel, requested_accelerations_mps2=[a1, a2],
                        requested_curvatures_per_m=[k1, k2], center_tube_radius_m=tube_radius,
                        points_xy=points.tolist(), yaw_rad=yaw.tolist(),
                        scope="same_joint_prediction_conservative_route_tube_and_continuous_OBB")
    return dict(verified=False, status="no_verified_witness_in_library", tested=tested)


def add_evidence(event, env, max_accel, search_witness=search):
    """只复核 route_grid 的真实空集，评分仍保留空面积和无效归一化。"""
    if (not event or event.get("reachability_backend") != "route_grid"
            or event.get("source_step") != env.current_step
            or event.get("dangerous_area_m2") != 0. or event.get("dangerous_solvable") is not False
            or env.attack_intent is None):
        return event
    event["avoidance_witness"] = search_witness(env, max_accel)
    # 空网格不能当成面积压缩 100%。保留原始面积曲线供复查。
    event["difficulty"] = float("nan")
    compression = event.get("response_space_compression", {})
    event["response_space_compression"] = {
        "curve": [dict(row, remaining_area_ratio=None, compression=None)
                  for row in compression.get("curve", [])],
        "minimum_remaining_area_ratio": None, "bottleneck_future_seconds": None,
    }
    return event


def apply_verdict(verdict, event, policy):
    """原安全检查成功后，仅允许同一零网格事件上有明确证书的情形。"""
    evidence = (event or {}).get("avoidance_witness", {})
    if (policy == "positive_area" and not verdict["allowed"] and event
            and event.get("dangerous_area_m2") == 0. and event.get("dangerous_solvable") is False
            and evidence.get("source_step") == event.get("source_step")
            and evidence.get("verified") is True and evidence.get("status") == "grid_empty_witness_feasible"):
        return dict(verdict, allowed=True, reason="grid_empty_witness_feasible",
                    relative_measurement_valid=False, relative_difficulty=None, avoidance_witness=evidence)
    return dict(verdict, avoidance_witness=evidence) if evidence else verdict
