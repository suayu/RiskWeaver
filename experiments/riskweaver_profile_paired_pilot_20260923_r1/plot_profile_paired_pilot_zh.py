"""为画像开关配对先导实验生成中文审阅图。"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


BASE = Path("/home2/zhaoyx/scenario-dreamer/experiments/riskweaver_profile_paired_pilot_20260923_r1")
SUMMARY = json.loads((BASE / "paired_summary.json").read_text(encoding="utf-8"))
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=FONT_PATH).get_name()]
plt.rcParams["axes.unicode_minus"] = False


groups = SUMMARY["groups"]
labels = ["画像关闭", "画像开启"]
keys = ["profile_off", "profile_on"]
colors = ["#4C78A8", "#F58518"]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

valid_d = [groups[key]["mean_valid_d"] for key in keys]
axes[0].bar(labels, valid_d, color=colors, width=0.58)
axes[0].set_title("有效观测上的平均危险度")
axes[0].set_ylabel("危险度 D")
for index, value in enumerate(valid_d):
    axes[0].text(index, value + 0.002, f"{value:.3f}", ha="center")

frames = [groups[key]["attack_executed_frames"] for key in keys]
axes[1].bar(labels, frames, color=colors, width=0.58)
axes[1].set_title("实际攻击执行覆盖")
axes[1].set_ylabel("累计攻击帧")
for index, value in enumerate(frames):
    axes[1].text(index, value + max(frames) * 0.02, str(value), ha="center")

x = np.arange(2)
coverage = [100 * groups[key]["valid_d_coverage"] for key in keys]
attack_scene_coverage = [100 * groups[key]["scenes_with_attack"] / groups[key]["scenario_count"] for key in keys]
axes[2].bar(x - 0.18, coverage, width=0.36, label="有效危险度覆盖率", color="#54A24B")
axes[2].bar(x + 0.18, attack_scene_coverage, width=0.36, label="实际攻击场景覆盖率", color="#E45756")
axes[2].set_xticks(x, labels)
axes[2].set_ylim(0, 112)
axes[2].set_ylabel("覆盖率（%）")
axes[2].set_title("覆盖率与缺失不可混为零")
axes[2].legend(fontsize=9, loc="upper center")

for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)

passed = SUMMARY["acceptance"]["passed"]
fig.suptitle(f"RiskWeaver 自车画像六场景配对先导实验（验收：{'通过' if passed else '未通过'}）", fontsize=15)
fig.text(
    0.5,
    0.01,
    "所有生成失败和无效危险度均计入覆盖率；D 的均值仅在有效观测上计算。",
    ha="center",
    fontsize=9,
    color="#555555",
)
fig.tight_layout(rect=[0, 0.05, 1, 0.92])
fig.savefig(BASE / "画像配对先导实验汇总.png", dpi=200)
plt.close(fig)


rows = SUMMARY["paired"]["rows"]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for row in rows:
    y = int(row["scenario_index"])
    off_d, on_d = row["off_d"], row["on_d"]
    if off_d is not None:
        ax.scatter(off_d, y, color=colors[0], s=60, zorder=3)
    if on_d is not None:
        ax.scatter(on_d, y, color=colors[1], s=60, zorder=3)
    if off_d is not None and on_d is not None:
        ax.plot([off_d, on_d], [y, y], color="#888888", linewidth=1.5, zorder=2)
    if off_d is None:
        # 将缺失提示放在绘图区内部，避免与场景名称及纵轴标签重叠。
        ax.text(0.001, y, "关闭组 D 无效", ha="left", va="center", color=colors[0], fontsize=8)
    if on_d is None:
        ax.text(0.001, y, "开启组 D 无效", ha="left", va="center", color=colors[1], fontsize=8)
ax.scatter([], [], color=colors[0], label="画像关闭")
ax.scatter([], [], color=colors[1], label="画像开启")
ax.set_yticks(range(len(rows)), [Path(row["scenario_source"]).name for row in rows])
ax.set_xlabel("场景危险度 D")
ax.set_title("逐场景配对危险度（缺失值不作零填充）")
ax.legend()
ax.grid(axis="x", alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(BASE / "画像配对逐场景危险度.png", dpi=200)
plt.close(fig)
