import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

root = Path("/home2/zhaoyx/scenario-dreamer")
out = root / "experiments/riskweaver_rl_profile_six_scene_summary_20260923"
summary = json.loads((out / "combined_summary.json").read_text(encoding="utf-8"))
rows = summary["paired"]["rows"]
font_zh = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
scenes = [Path(row["scenario_source"]).stem for row in rows]
x = np.arange(len(scenes))
off_frames = [row["off_attack_frames"] for row in rows]
on_frames = [row["on_attack_frames"] for row in rows]
off_d = [np.nan if row["off_scene_d"] is None else row["off_scene_d"] for row in rows]
on_d = [np.nan if row["on_scene_d"] is None else row["on_scene_d"] for row in rows]

def draw(path, chinese):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, off_frames, width, label="画像关闭" if chinese else "Profile off")
    axes[0].bar(x + width / 2, on_frames, width, label="画像开启" if chinese else "Profile on")
    axes[0].set_ylabel("实际攻击帧数" if chinese else "Executed attack frames", fontproperties=font_zh if chinese else None)
    axes[0].set_title("GPUDrive 六场画像校准：攻击执行覆盖" if chinese else "GPUDrive six-scene profile calibration: attack execution", fontproperties=font_zh if chinese else None)
    axes[0].set_xticks(x, scenes)
    axes[0].legend(prop=font_zh if chinese else None)
    axes[0].grid(axis="y", alpha=.25)
    axes[1].plot(x, off_d, "o-", label="画像关闭" if chinese else "Profile off")
    axes[1].plot(x, on_d, "o-", label="画像开启" if chinese else "Profile on")
    axes[1].set_ylabel("场景危险度 D" if chinese else "Scenario danger D", fontproperties=font_zh if chinese else None)
    axes[1].set_xlabel("冻结场景" if chinese else "Frozen scenario", fontproperties=font_zh if chinese else None)
    axes[1].set_title("有效观测窗口加权危险度（缺失值不按 0 处理）" if chinese else "Window-weighted danger (missing values are not zero)", fontproperties=font_zh if chinese else None)
    axes[1].set_xticks(x, scenes)
    axes[1].legend(prop=font_zh if chinese else None)
    axes[1].grid(alpha=.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)

draw(out / "六场画像校准结果.png", True)
draw(out / "six_scene_profile_calibration_en.png", False)

video_out = out / "videos"
video_out.mkdir(exist_ok=True)
group_bases = summary["source_batches"]
for group, bases in group_bases.items():
    group_out = video_out / group
    group_out.mkdir(exist_ok=True)
    for base_text in bases:
        for result_path in Path(base_text).joinpath("movies").rglob("attempt_result.json"):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            source = Path(result["scenario_source"]).stem
            movie = Path(result["video_path"])
            if movie.exists():
                shutil.copy2(movie, group_out / f"{source}.mp4")
