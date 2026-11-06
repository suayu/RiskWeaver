"""根据一次真实闭环运行记录绘制 LLM 联合引导结果。"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attack-step", type=int, required=True)
    parser.add_argument("--target-id", type=int, required=True)
    parser.add_argument("--anchors", type=json.loads, required=True)
    args = parser.parse_args()

    with open(args.scenario, "rb") as scenario_file:
        scenario = pickle.load(scenario_file)
    with open(args.steps, "r", encoding="utf-8") as step_file:
        steps = json.load(step_file)

    ego = np.asarray(steps["ego_vehicle"], dtype=np.float32)
    target = np.asarray(steps[f"agent_{args.target_id}"], dtype=np.float32)
    anchors = np.asarray(args.anchors, dtype=np.float32)
    start = int(args.attack_step)
    times = (np.arange(start, len(target), dtype=np.float32) - start + 1.0) * 0.1
    desired = np.column_stack(
        [np.interp(times, np.arange(len(anchors)), anchors[:, axis]) for axis in range(2)]
    )
    anchor_error = np.linalg.norm(target[start:, :2] - desired, axis=-1)
    post_attack_distance = np.linalg.norm(target[start:, :2] - ego[start:, :2], axis=-1)

    figure, axis = plt.subplots(figsize=(8, 9))
    for lane in np.asarray(scenario["lanes"]):
        axis.plot(lane[:, 0], lane[:, 1], color="0.82", linewidth=0.8, zorder=0)
    axis.plot(
        np.asarray(scenario["route"])[:, 0],
        np.asarray(scenario["route"])[:, 1],
        color="0.45",
        linestyle="--",
        linewidth=1.2,
        label="Ego route",
    )
    axis.plot(ego[:, 0], ego[:, 1], color="black", linewidth=2.4, label="Ego executed")
    axis.plot(
        target[:, 0], target[:, 1], color="tab:orange", linewidth=2.4,
        label=f"Target {args.target_id} executed",
    )
    axis.plot(
        anchors[:, 0], anchors[:, 1], "o--", color="tab:red", linewidth=1.8,
        label="LLM anchors",
    )
    axis.scatter(
        [ego[start, 0], target[start, 0]],
        [ego[start, 1], target[start, 1]],
        marker="*", s=150, color=["black", "tab:orange"], zorder=5,
        label=f"Guidance starts at step {start}",
    )
    summary = (
        f"post-attack min center distance: {post_attack_distance.min():.2f} m\n"
        f"anchor ADE over executed frames: {anchor_error.mean():.2f} m\n"
        "losses: collision + route + TTC + LLM anchor"
    )
    axis.text(
        0.02, 0.98, summary, transform=axis.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )
    visible = np.concatenate([ego[:, :2], target[:, :2], anchors], axis=0)
    lower = visible.min(axis=0) - 8.0
    upper = visible.max(axis=0) + 8.0
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Global X (m)")
    axis.set_ylabel("Global Y (m)")
    axis.set_title("Scenario Dreamer + Safe-Sim: LLM Joint Guidance")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(json.dumps({
        "output": str(output),
        "post_attack_min_center_distance_m": float(post_attack_distance.min()),
        "executed_anchor_ade_m": float(anchor_error.mean()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
