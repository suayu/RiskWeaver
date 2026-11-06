"""汇总 RiskWeaver、SAFE-SIM 与 Scenario Dreamer 的冻结配对实验。"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


METRICS = {
    "min_ttc_s": "min_ttc_s",
    "max_ea_mps2": "max_evasive_acceleration_mps2",
    "progress": "progress",
    "danger": "scenario_danger_score",
    "ability": "autonomous_driving_ability_score",
}


def load_episodes(path):
    """读取统一结果或 RiskWeaver 多场结果，不修改缺失值。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if episodes is None:
        episodes = payload.get("results")
    if not isinstance(episodes, list):
        raise ValueError(f"结果缺少 episodes/results 数组：{path}")
    return episodes


def scenario_key(row):
    source = row.get("scenario_source") or row.get("scenario_file")
    if source:
        return Path(str(source)).name
    index = row.get("scenario_index")
    if index is None:
        raise ValueError("场景记录缺少来源和索引")
    return f"index:{int(index)}"


def finite(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None:
            value = float(value)
            return value if np.isfinite(value) else None
    return None


def normalize_episode(method, row):
    danger = finite(row, "scenario_danger_score", "danger")
    danger_valid = bool(row.get("danger_valid", danger is not None))
    if not danger_valid:
        danger = None
    ability = finite(row, "autonomous_driving_ability_score", "ability_score")
    ability_valid = bool(row.get("ability_valid", ability is not None))
    if not ability_valid:
        ability = None
    return {
        "method": method,
        "scenario": scenario_key(row),
        "collision": int(bool(row.get("collision", False))),
        "completed": int(bool(row.get("completed", False))),
        "off_route": int(bool(row.get("off_route", False))),
        "progress": finite(row, "progress"),
        "min_ttc_s": finite(row, "min_ttc_s", "min_ttc"),
        "max_ea_mps2": finite(
            row, "max_evasive_acceleration_mps2", "max_ea_mps2"
        ),
        "danger": danger,
        "ability": ability,
        "attack_plan_count": int(row.get("attack_plan_count", 0) or 0),
        "attack_executed_frames": int(
            row.get("attack_executed_frames", row.get("actual_attack_frames", 0)) or 0
        ),
        "planner_service_failures": int(
            row.get("planner_service_failure_count", 0) or 0
        ),
        "background_collision_frames": int(
            row.get("feasibility", {}).get(
                "background_collision_frames",
                row.get("background_collision_frames_crosscheck", 0),
            ) or 0
        ),
        "background_static_collision_frames": int(
            row.get("feasibility", {}).get(
                "background_static_collision_frames", 0
            ) or 0
        ),
    }


def mean(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def aggregate(rows):
    return {
        "scenario_count": len(rows),
        "collision_rate": mean(row["collision"] for row in rows),
        "completion_rate": mean(row["completed"] for row in rows),
        "off_route_rate": mean(row["off_route"] for row in rows),
        "mean_progress": mean(row["progress"] for row in rows),
        "mean_min_ttc_s": mean(row["min_ttc_s"] for row in rows),
        "mean_episode_max_ea_mps2": mean(row["max_ea_mps2"] for row in rows),
        "danger_valid_count": sum(row["danger"] is not None for row in rows),
        "danger_valid_coverage": mean(row["danger"] is not None for row in rows),
        "mean_valid_danger": mean(row["danger"] for row in rows),
        "ability_valid_count": sum(row["ability"] is not None for row in rows),
        "mean_valid_ability": mean(row["ability"] for row in rows),
        "attack_plan_count": sum(row["attack_plan_count"] for row in rows),
        "attack_executed_frames": sum(row["attack_executed_frames"] for row in rows),
        "planner_service_failures": sum(row["planner_service_failures"] for row in rows),
        "background_collision_frames": sum(
            row["background_collision_frames"] for row in rows
        ),
        "background_static_collision_frames": sum(
            row["background_static_collision_frames"] for row in rows
        ),
    }


def paired_bootstrap(rows_a, rows_b, metric, seed=42, samples=10000):
    """仅使用两组共同有效的同一场景，返回 A-B 配对差区间。"""
    a = {row["scenario"]: row[metric] for row in rows_a if row[metric] is not None}
    b = {row["scenario"]: row[metric] for row in rows_b if row[metric] is not None}
    common = sorted(set(a) & set(b))
    if not common:
        return {"paired_count": 0, "mean_difference": None, "ci95": None}
    differences = np.asarray([a[key] - b[key] for key in common], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    distribution = differences[indices].mean(axis=1)
    return {
        "paired_count": len(common),
        "mean_difference": float(differences.mean()),
        "ci95": [float(x) for x in np.quantile(distribution, [0.025, 0.975])],
    }


def plot_summary(aggregates, output, language):
    labels = list(aggregates)
    zh = language == "zh"
    if zh:
        # 显式加载常见 CJK 字体，防止无图形界面服务器生成方框字。
        candidates = (
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
        )
        font_path = next((path for path in candidates if path.exists()), None)
        if font_path is None:
            raise RuntimeError("找不到可用于中文图的 CJK 字体")
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=str(font_path)
        ).get_name()
        plt.rcParams["axes.unicode_minus"] = False
    titles = (
        ["最小 TTC↓", "最大规避加速度↑", "碰撞率↑", "进度"]
        if zh else
        ["Minimum TTC↓", "Maximum EA↑", "Collision rate↑", "Progress"]
    )
    keys = ["mean_min_ttc_s", "mean_episode_max_ea_mps2", "collision_rate", "mean_progress"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.1))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for axis, title, key in zip(axes, titles, keys):
        values = [aggregates[label].get(key) for label in labels]
        plotted = [np.nan if value is None else value for value in values]
        axis.bar(labels, plotted, color=colors[:len(labels)])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            if value is not None:
                axis.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(method_paths, output_dir, bootstrap_samples=10000):
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = {
        method: [normalize_episode(method, row) for row in load_episodes(path)]
        for method, path in method_paths.items()
    }
    scenario_sets = {method: {row["scenario"] for row in rows}
                     for method, rows in normalized.items()}
    if len({frozenset(items) for items in scenario_sets.values()}) != 1:
        raise ValueError("方法间场景集合不一致，拒绝生成伪配对比较")
    aggregates = {method: aggregate(rows) for method, rows in normalized.items()}
    comparisons = {}
    methods = list(normalized)
    for index, method_a in enumerate(methods):
        for method_b in methods[index + 1:]:
            comparisons[f"{method_a}_minus_{method_b}"] = {
                metric: paired_bootstrap(
                    normalized[method_a], normalized[method_b], metric,
                    samples=bootstrap_samples,
                )
                for metric in ("min_ttc_s", "max_ea_mps2", "progress", "danger", "ability")
            }
    report = {
        "methods": {name: str(path) for name, path in method_paths.items()},
        "aggregate": aggregates,
        "paired_comparisons": comparisons,
        "missing_semantics": "null values are excluded and reported through valid coverage; never imputed as zero",
    }
    (output_dir / "method_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    rows = [row for method_rows in normalized.values() for row in method_rows]
    with (output_dir / "method_comparison_episodes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(aggregates, output_dir / "方法对比_中文.png", "zh")
    plot_summary(aggregates, output_dir / "method_comparison_en.png", "en")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", action="append", required=True,
        help="METHOD=/absolute/path/to/results.json，可重复传入",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    method_paths = {}
    for item in args.method:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"无效 --method 参数：{item}")
        method_paths[name] = Path(path)
    summarize(method_paths, args.output_dir, args.bootstrap_samples)


if __name__ == "__main__":
    main()
