"""端到端评估 LLM、联合扩散引导与静态障碍物调度的实际闭环表现。"""

import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from hydra import compose, initialize

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from policies.idm_policy import IDMPolicy
from scenario_generator import AdversarialScenarioGenerator
from simulator import Simulator
from utils.collision_helpers import compute_collision_states_one_scene
from utils.data_helpers import modify_agent_states
from utils.viz import generate_video


FULL_GUIDANCE_FUNCTIONS = ["scenario_collision", "route", "scenario_ttc", "llm_anchor"]
TEST_INSTRUCTION = [
    "生成一个可避免但具有挑战性的自动驾驶安全测试场景。"
    "在道路几何、现有车辆和安全约束允许时，优先用抽象静态障碍物收窄或阻断自车前方道路，"
    "并协调一辆交通参与者的对抗轨迹；若不具备物理可行性，则不放置障碍物且不攻击。"
]


def set_seed(seed):
    """固定所有随机源，使不同运行的扩散采样和场景选择可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_scenarios(env, requested_count):
    """优先选择同向前车或相邻车道车辆，避免只按距离选到对向交通。"""
    interaction_candidates = []
    fallback_candidates = []
    for index, scenario_path in enumerate(env.test_files):
        with open(scenario_path, "rb") as scenario_file:
            scenario = pickle.load(scenario_file)
        states = np.asarray(scenario["agents"][:, 0], dtype=np.float32)
        types = np.asarray(scenario["agent_types"], dtype=np.float32)
        ego = states[-1]
        vehicles = states[:-1][types[:-1, 1] == 1]
        if len(vehicles) == 0:
            continue
        distances = np.linalg.norm(vehicles[:, :2] - ego[None, :2], axis=-1)
        nearby = distances[(distances >= 5.0) & (distances <= 45.0)]
        if len(nearby):
            fallback_candidates.append((float(nearby.min()), index))

        ego_velocity = ego[2:4]
        ego_speed = float(np.linalg.norm(ego_velocity))
        if ego_speed < 1.0:
            continue
        ego_forward = ego_velocity / ego_speed
        for vehicle in vehicles:
            vehicle_velocity = vehicle[2:4]
            vehicle_speed = float(np.linalg.norm(vehicle_velocity))
            if vehicle_speed < 1.0:
                continue
            direction_similarity = float(np.dot(ego_forward, vehicle_velocity / vehicle_speed))
            relative_position = vehicle[:2] - ego[:2]
            longitudinal = float(np.dot(relative_position, ego_forward))
            lateral = float(abs(
                ego_forward[0] * relative_position[1] - ego_forward[1] * relative_position[0]
            ))
            # 同向前方、可在三秒窗口内接近的车辆更适合验证制动或切入攻击。
            if direction_similarity < 0.5 or not 5.0 <= longitudinal <= 35.0 or lateral > 9.0:
                continue
            score = abs(longitudinal - 16.0) + 1.5 * abs(lateral - 3.5)
            interaction_candidates.append((float(score), index))
            break
    candidates = interaction_candidates if len(interaction_candidates) >= requested_count else fallback_candidates
    if len(candidates) < requested_count:
        raise RuntimeError("可用于端到端评估的近邻车辆场景数量不足")
    return [index for _, index in sorted(candidates)[:requested_count]]


def min_center_distance_and_ttc(ego_state, agent_states, active_mask):
    """计算当前帧自车与活动背景车的最小中心距离和常速近似 TTC。"""
    active_states = np.asarray(agent_states)[np.asarray(active_mask, dtype=bool)]
    if len(active_states) == 0:
        return float("inf"), float("inf")
    relative_position = active_states[:, :2] - ego_state[None, :2]
    distances = np.linalg.norm(relative_position, axis=-1)
    relative_velocity = active_states[:, 2:4] - ego_state[None, 2:4]
    speed_squared = np.sum(relative_velocity * relative_velocity, axis=-1)
    ttc = np.full(len(active_states), np.inf, dtype=np.float64)
    moving = speed_squared > 1e-6
    ttc[moving] = -np.sum(
        relative_position[moving] * relative_velocity[moving], axis=-1
    ) / speed_squared[moving]
    ttc[(ttc <= 0.0) | ~np.isfinite(ttc)] = np.inf
    return float(distances.min()), float(ttc.min())


def background_collision_count(agent_states, active_mask):
    """统计活动背景交通参与者的几何碰撞数量，不把自车计入其中。"""
    active_states = np.asarray(agent_states)[np.asarray(active_mask, dtype=bool)]
    if len(active_states) < 2:
        return 0
    colliding = compute_collision_states_one_scene(modify_agent_states(active_states))
    return int(np.count_nonzero(colliding))


def draw_summary(output_path, scenario, record):
    """绘制真实闭环执行轨迹、LLM 锚点和最终静态障碍物。"""
    figure, axis = plt.subplots(figsize=(9, 9))
    for lane in np.asarray(scenario["lanes"]):
        axis.plot(lane[:, 0], lane[:, 1], color="0.82", linewidth=0.8, zorder=0)
    axis.plot(
        np.asarray(scenario["route"])[:, 0],
        np.asarray(scenario["route"])[:, 1],
        color="0.45", linestyle="--", linewidth=1.2, label="自车参考路线",
    )
    ego_trace = np.asarray(record["ego_trace"], dtype=np.float32)
    axis.plot(ego_trace[:, 0], ego_trace[:, 1], color="black", linewidth=2.4, label="自车实际轨迹")
    for agent_id, trace in record["agent_traces"].items():
        trace = np.asarray(trace, dtype=np.float32)
        color = "tab:orange" if int(agent_id) == record.get("attack_target_id") else "tab:blue"
        alpha = 1.0 if int(agent_id) == record.get("attack_target_id") else 0.35
        label = "LLM 目标车实际轨迹" if int(agent_id) == record.get("attack_target_id") else None
        axis.plot(trace[:, 0], trace[:, 1], color=color, alpha=alpha, linewidth=1.8, label=label)
    if record.get("anchors"):
        anchors = np.asarray(record["anchors"], dtype=np.float32)
        axis.plot(anchors[:, 0], anchors[:, 1], "o--", color="tab:red", linewidth=2.0, label="LLM 锚点")
    for obstacle in record["static_obstacles"]:
        center = obstacle["center"]
        rectangle = patches.Rectangle(
            (center[0] - obstacle["width"] / 2.0, center[1] - obstacle["length"] / 2.0),
            obstacle["width"], obstacle["length"],
            facecolor="tab:red", edgecolor="darkred", alpha=0.6, zorder=5,
        )
        rectangle.set_transform(
            transforms.Affine2D().rotate_deg_around(
                center[0], center[1], math.degrees(obstacle["yaw"]) - 90.0
            ) + axis.transData
        )
        axis.add_patch(rectangle)
    annotation = (
        f"场景 {record['scenario_index']}，执行 {record['executed_steps']} 帧\n"
        f"最小中心距离: {record['min_center_distance_m']:.2f} m\n"
        f"最小常速 TTC: {record['min_ttc_s']:.2f} s\n"
        f"引导: {' + '.join(record['guidance_functions'])}"
    )
    axis.text(
        0.02, 0.98, annotation, transform=axis.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("LLM + Safe-Sim 联合对抗轨迹评估")
    axis.set_xlabel("全局 X（米）")
    axis.set_ylabel("全局 Y（米）")
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def draw_attack_reason_comparison(output_path, records):
    """以逐场景攻击结果和最新 LLM 原因建立可审计的对照图。"""
    figure, (axis, text_axis) = plt.subplots(
        1, 2, figsize=(16, max(7, 0.38 * len(records) + 2)),
        gridspec_kw={"width_ratios": [1.0, 2.4]},
    )
    labels, colors, rows = [], [], []
    for record in records:
        reasons = record.get("attack_reasons", [])
        accepted = [item for item in reasons if item.get("accepted")]
        latest = (accepted or reasons or [{}])[-1]
        if record["collision"]:
            outcome, color = "自车碰撞", "tab:red"
        elif record["attack_accepted"] and record["static_obstacle_count"]:
            outcome, color = "联合攻击", "tab:purple"
        elif record["attack_accepted"]:
            outcome, color = "动态攻击", "tab:orange"
        else:
            outcome, color = "未执行攻击", "0.6"
        labels.append(str(record["scenario_index"]))
        colors.append(color)
        reason = str(latest.get("reason") or "未返回可用攻击原因").replace("\n", " ")
        rows.append(
            f"#{record['scenario_index']:03d} | {outcome} | "
            f"目标 {latest.get('target_id', '-')} | {latest.get('strategy', '-')}\n  {reason[:110]}"
        )
    positions = np.arange(len(records))
    axis.barh(positions, np.ones(len(records)), color=colors)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("攻击/结果状态")
    axis.set_ylabel("场景索引")
    axis.set_xlim(0, 1.15)
    axis.set_xticks([])
    axis.set_title("攻击执行结果（红：自车碰撞；紫：联合攻击）")
    axis.grid(axis="y", alpha=0.2)
    text_axis.axis("off")
    text_axis.set_title("LLM 返回的发起攻击原因（最近有效计划）", loc="left")
    text_axis.text(0.0, 0.99, "\n\n".join(rows), va="top", fontsize=8, wrap=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_episode(env, policy, generator, scenario_index, steps, movie_path=None):
    """执行单个闭环场景并收集规划、风险、耗时和轨迹证据。"""
    obs = env.reset(scenario_index)
    generator.reset_episode_stats()
    policy.reset(obs)
    diffusion_times = []
    min_distance = float("inf")
    min_ttc = float("inf")
    guidance_active_frames = 0
    background_collision_frames = 0
    background_collision_agents = 0
    candidate_route_frames = 0
    max_candidate_route_count = 0
    attack_target_id = None
    anchors = None
    terminal_info = {"collision": False, "off_route": False, "completed": False}
    for _ in range(steps):
        generator.step(env, env.current_step)
        start_time = time.perf_counter()
        joint = env.prepare_background_traffic()
        diffusion_times.append(time.perf_counter() - start_time)
        candidate_count = len(getattr(env, "latest_diffusion_candidate_trajectories", []))
        candidate_route_frames += int(candidate_count > 0)
        max_candidate_route_count = max(max_candidate_route_count, candidate_count)
        metadata = joint.metadata
        if metadata.get("adversarial_guidance_active"):
            guidance_active_frames += 1
        if env.attack_intent is not None:
            attack_target_id = env.attack_intent["target_id"]
            anchors = env.attack_intent["anchors"]
        if movie_path is not None:
            # 每场景独立目录，完整保存 E1 闭环帧，避免跨场景混帧。
            env.render_state(name=f"scenario_{scenario_index:03d}", movie_path=str(movie_path))
        action = policy.act(obs)
        # 使用本帧联合筛选的攻击目标轨迹，保证红色最终路线与半透明候选路线同源。
        selected_trajectory = env.get_attack_target_prediction()
        generator.set_diffusion_trajectory(selected_trajectory)
        obs, terminated, info = env.step(action, anchors, selected_trajectory)
        distance, ttc = min_center_distance_and_ttc(
            env.ego_state, env.data_dict["agent"][-1], env.agent_active
        )
        min_distance = min(min_distance, distance)
        min_ttc = min(min_ttc, ttc)
        colliding_agents = background_collision_count(env.data_dict["agent"][-1], env.agent_active)
        if colliding_agents:
            background_collision_frames += 1
            background_collision_agents += colliding_agents
        if terminated:
            terminal_info = info
            break

    ego_trace = [state[0].tolist() for state in env.data_dict["ego"]]
    scene_difficulty = generator.episode_difficulty()
    route_progress = min(1.0, len(diffusion_times) / max(steps, 1))
    ability_alpha = float(env.cfg.sim.evaluation.difficulty_control.ability_progress_alpha)
    ego_ability_score = 100.0 * scene_difficulty * ((0.0 if terminal_info.get("collision", False) else 1.0) + ability_alpha * route_progress)
    agent_traces = {
        str(agent_id): [frame[agent_id].tolist() for frame in env.data_dict["agent"]]
        for agent_id in range(len(env.data_dict["agent"][0]))
    }
    video_path = generate_video(f"scenario_{scenario_index:03d}", str(movie_path), delete_images=False) if movie_path is not None else None
    return {
        "scenario_index": int(scenario_index),
        "scenario_id": str(env.current_scene_id),
        "executed_steps": len(diffusion_times),
        "llm_call_count": int(generator._llm_call_times),
        "attack_target_id": attack_target_id,
        "attack_accepted": attack_target_id is not None,
        "static_obstacle_count": len(env.get_static_obstacles()),
        "static_obstacles": env.get_static_obstacles(),
        "guidance_functions": list(env.diffusion_controller.active_guidance_functions),
        "guidance_active_frames": int(guidance_active_frames),
        "mean_diffusion_seconds": float(np.mean(diffusion_times)),
        "p95_diffusion_seconds": float(np.percentile(diffusion_times, 95)),
        "min_center_distance_m": float(min_distance),
        "min_ttc_s": float(min_ttc),
        "collision": bool(terminal_info.get("collision", False)),
        "off_route": bool(terminal_info.get("off_route", False)),
        "completed": bool(terminal_info.get("completed", False)),
        "background_collision_frames": int(background_collision_frames),
        "background_colliding_agent_count": int(background_collision_agents),
        "candidate_route_frames": int(candidate_route_frames),
        "max_candidate_route_count": int(max_candidate_route_count),
        "attack_window_difficulties": list(generator._episode_attack_difficulties),
        "scene_difficulty": float(scene_difficulty),
        "ego_ability_score": float(ego_ability_score),
        "route_progress": float(route_progress),
        "attack_reasons": list(generator._episode_attack_reasons),
        "video_path": video_path,
        "anchors": anchors,
        "ego_trace": ego_trace,
        "agent_traces": agent_traces,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=2)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--output-dir", default="movies/full_adversarial_evaluation")
    parser.add_argument("--save-videos", action="store_true", help="逐场景保存完整 PNG 序列和 MP4")
    parser.add_argument(
        "--show-diffusion-candidates", action="store_true",
        help="在视频中半透明绘制攻击车辆的全部扩散候选路线",
    )
    parser.add_argument("--scenario-indices", default="", help="逗号分隔的固定场景编号")
    parser.add_argument(
        "--sim-override",
        action="append",
        default=[],
        help="可重复指定 Hydra sim 覆盖项，例如 sim.traffic_model.num_samples=8",
    )
    parser.add_argument(
        "--attack-mode",
        choices=["trajectory_only", "obstacle_only", "joint"],
        default="joint",
    )
    args = parser.parse_args()
    overrides = [
        "sim.visualize=false",
        f"sim.steps={args.steps}",
        "sim.traffic_model.guidance.mode=llm_joint",
        f"sim.llm.attack_mode={args.attack_mode}",
        "sim.traffic_model.iterative_adversarial.profile_enabled=true",
        "sim.traffic_model.iterative_adversarial.escalation_enabled=true",
        "sim.traffic_model.iterative_adversarial.full_method_enabled=true",
    ]
    if args.show_diffusion_candidates:
        overrides.append("sim.visualization.show_diffusion_candidates=true")
    overrides.extend(args.sim_override)
    with initialize(version_base=None, config_path="../cfgs"):
        cfg = compose(config_name="config", overrides=overrides)
    set_seed(int(cfg.sim.seed))
    env = Simulator(cfg)
    if env.diffusion_controller.active_guidance_functions != FULL_GUIDANCE_FUNCTIONS:
        raise RuntimeError(
            f"未启用全部联合引导项：{env.diffusion_controller.active_guidance_functions}"
        )
    if args.attack_mode in {"obstacle_only", "joint"} and not env.llm_planner.use_obstacles:
        raise RuntimeError("静态障碍物调度开关未生效")
    policy = IDMPolicy(cfg, env)
    if args.attack_mode == "obstacle_only":
        test_instruction = [
            "在自车前方的路线中心线附近放置一个临时、可避免的静态障碍物，"
            "优先测试自车减速或绕行能力；无需协调任何交通参与者。"
        ]
    else:
        test_instruction = TEST_INSTRUCTION
    generator = AdversarialScenarioGenerator(cfg, test_instruction, llm_planner=env.llm_planner)
    if args.scenario_indices:
        scenario_indices = [int(item) for item in args.scenario_indices.split(",") if item.strip()]
        if not scenario_indices or any(index < 0 or index >= len(env.test_files) for index in scenario_indices):
            raise ValueError("指定的场景编号为空或超出数据集范围")
    else:
        scenario_indices = choose_scenarios(env, args.scenarios)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        run_episode(env, policy, generator, index, args.steps, output_dir / f"scenario_{index:03d}" if args.save_videos else None)
        for index in scenario_indices
    ]
    with open(env.test_files[records[0]["scenario_index"]], "rb") as scenario_file:
        representative_scenario = pickle.load(scenario_file)
    draw_summary(output_dir / "representative_trajectory.png", representative_scenario, records[0])
    reason_visualization = output_dir / "attack_reason_comparison.png"
    draw_attack_reason_comparison(reason_visualization, records)
    summary = {
        "configuration": {
            "guidance_mode": env.diffusion_controller.guidance_mode,
            "guidance_functions": env.diffusion_controller.active_guidance_functions,
            "llm_obstacles_enabled": env.llm_planner.use_obstacles,
            "llm_attack_mode": env.llm_planner.attack_mode,
            "scenario_indices": scenario_indices,
            "steps_per_scenario": args.steps,
            "iterative_adversarial": {"profile": True, "escalation": True, "full_method": True},
            "save_videos": bool(args.save_videos),
            "show_diffusion_candidates": bool(args.show_diffusion_candidates),
        },
        "aggregate": {
            "attack_acceptance_rate": float(np.mean([record["attack_accepted"] for record in records])),
            "obstacle_deployment_rate": float(np.mean([record["static_obstacle_count"] > 0 for record in records])),
            "collision_rate": float(np.mean([record["collision"] for record in records])),
            "background_collision_episode_rate": float(np.mean([
                record["background_collision_frames"] > 0 for record in records
            ])),
            "mean_background_collision_frames": float(np.mean([
                record["background_collision_frames"] for record in records
            ])),
            "mean_background_colliding_agent_count": float(np.mean([
                record["background_colliding_agent_count"] for record in records
            ])),
            "mean_min_center_distance_m": float(np.mean([record["min_center_distance_m"] for record in records])),
            "mean_min_ttc_s": float(np.mean([record["min_ttc_s"] for record in records])),
            "mean_diffusion_seconds": float(np.mean([record["mean_diffusion_seconds"] for record in records])),
            "mean_candidate_route_frames": float(np.mean([record["candidate_route_frames"] for record in records])),
            "mean_max_candidate_route_count": float(np.mean([record["max_candidate_route_count"] for record in records])),
        },
        "episodes": records,
        "representative_visualization": str(output_dir / "representative_trajectory.png"),
        "attack_reason_comparison_visualization": str(reason_visualization),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
    # 控制台仅输出核心指标；完整轨迹保留在 summary.json，避免日志被大量状态数据淹没。
    concise_episodes = [
        {
            key: record[key]
            for key in (
                "scenario_index", "scenario_id", "executed_steps", "llm_call_count",
                "attack_target_id", "attack_accepted", "static_obstacle_count",
                "guidance_active_frames", "mean_diffusion_seconds", "p95_diffusion_seconds",
                "min_center_distance_m", "min_ttc_s", "collision", "off_route", "completed",
                "background_collision_frames", "background_colliding_agent_count",
                "candidate_route_frames", "max_candidate_route_count",
            )
        }
        for record in records
    ]
    print(json.dumps({
        "configuration": summary["configuration"],
        "aggregate": summary["aggregate"],
        "episodes": concise_episodes,
        "representative_visualization": summary["representative_visualization"],
        "attack_reason_comparison_visualization": summary["attack_reason_comparison_visualization"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
