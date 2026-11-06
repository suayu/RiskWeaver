"""在真实 Safe-Sim 权重上对比无引导轨迹与 LLM 锚点引导轨迹。"""

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
from hydra import compose, initialize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import Simulator


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _interpolate_anchors(anchors, horizon, dt, interval):
    anchor_times = np.arange(len(anchors), dtype=np.float32) * interval
    prediction_times = (np.arange(horizon, dtype=np.float32) + 1.0) * dt
    valid = prediction_times <= anchor_times[-1] + 1e-6
    targets = np.column_stack(
        [np.interp(prediction_times[valid], anchor_times, anchors[:, axis]) for axis in range(2)]
    )
    return targets.astype(np.float32), valid


def main():
    overrides = [
        "sim.visualize=false",
        "sim.traffic_model.num_samples=8",
        "sim.traffic_model.sample_step=5",
        "sim.traffic_model.anchor_guidance.enabled=true",
        "sim.traffic_model.anchor_guidance.strength=0.2",
        "sim.traffic_model.anchor_guidance.inner_lr=0.2",
        "sim.traffic_model.anchor_guidance.scale_grad_by_std=false",
    ]
    with initialize(version_base=None, config_path="../cfgs"):
        cfg = compose(config_name="config", overrides=overrides)

    env = Simulator(cfg)
    env.reset(0)
    frame = env.get_scenario_frame(env.diffusion_controller.adapter.history_frames)
    active_ids = frame.agent_ids[frame.active_mask]
    if len(active_ids) == 0:
        raise RuntimeError("验证场景没有活动交通参与者")
    target_id = int(active_ids[0])
    target_state = frame.states_global[target_id]
    speed = max(float(np.linalg.norm(target_state[2:4])), 3.0)
    yaw = float(target_state[4])
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    right = np.array([math.sin(yaw), -math.cos(yaw)], dtype=np.float32)
    anchor_times = np.arange(4, dtype=np.float32)
    anchors = np.stack(
        [
            target_state[:2] + speed * time * forward + 1.5 * (time / 3.0) * right
            for time in anchor_times
        ]
    ).astype(np.float32)
    _set_seed(2026)
    unguided = env.diffusion_controller.predict(frame, attack_intent=None)
    env.set_attack_intent(
        target_id=target_id,
        anchors=anchors.tolist(),
        strategy="cut_in",
    )
    _set_seed(2026)
    guided = env.prepare_background_traffic()

    target_row = int(np.flatnonzero(guided.agent_ids == target_id)[0])
    desired, valid = _interpolate_anchors(
        anchors,
        env.diffusion_controller.prediction_horizon,
        frame.dt,
        env.diffusion_controller.anchor_interval_seconds,
    )
    unguided_error = np.linalg.norm(
        unguided.positions_global[target_row, valid] - desired,
        axis=-1,
    )
    guided_error = np.linalg.norm(
        guided.positions_global[target_row, valid] - desired,
        axis=-1,
    )
    other_rows = np.flatnonzero(guided.agent_ids != target_id)
    if len(other_rows):
        other_max_difference = float(
            np.max(
                np.abs(
                    guided.positions_global[other_rows]
                    - unguided.positions_global[other_rows]
                )
            )
        )
    else:
        other_max_difference = 0.0

    output_path = Path("movies/anchor_guidance_validation.png").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 7))
    for lane in frame.lanes_global:
        axis.plot(lane[:, 0], lane[:, 1], color="0.75", linewidth=1.0)
    axis.plot(anchors[:, 0], anchors[:, 1], "ko--", label="LLM anchors")
    axis.plot(
        unguided.positions_global[target_row, :, 0],
        unguided.positions_global[target_row, :, 1],
        color="tab:blue",
        linewidth=2.0,
        label=f"Unguided (ADE={unguided_error.mean():.2f} m)",
    )
    axis.plot(
        guided.positions_global[target_row, :, 0],
        guided.positions_global[target_row, :, 1],
        color="tab:orange",
        linewidth=2.5,
        label=f"Anchor guided (ADE={guided_error.mean():.2f} m)",
    )
    axis.scatter(
        target_state[0],
        target_state[1],
        marker="*",
        s=180,
        color="tab:red",
        label=f"Target agent {target_id}",
        zorder=5,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Safe-Sim LLM Anchor Guidance Validation")
    axis.set_xlabel("Global X (m)")
    axis.set_ylabel("Global Y (m)")
    visible_points = np.concatenate(
        [
            anchors,
            unguided.positions_global[target_row],
            guided.positions_global[target_row],
        ],
        axis=0,
    )
    minimum = visible_points.min(axis=0) - 8.0
    maximum = visible_points.max(axis=0) + 8.0
    axis.set_xlim(minimum[0], maximum[0])
    axis.set_ylim(minimum[1], maximum[1])
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    result = {
        "target_id": target_id,
        "valid_guidance_steps": int(valid.sum()),
        "unguided_ade_m": float(unguided_error.mean()),
        "guided_ade_m": float(guided_error.mean()),
        "improvement_ratio": float(1.0 - guided_error.mean() / unguided_error.mean()),
        "other_agents_max_difference_m": other_max_difference,
        "guidance_metadata": guided.metadata,
        "visualization": str(output_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not guided_error.mean() < unguided_error.mean():
        raise AssertionError("锚点 guidance 未降低攻击车轨迹误差")
    if other_max_difference > 1e-4:
        raise AssertionError("锚点 guidance 改变了非目标交通参与者的轨迹")


if __name__ == "__main__":
    main()
