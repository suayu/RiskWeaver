"""汇总 RiskWeaver 画像开关配对先导实验，保留无效危险度与全部失败。"""

import csv
import json
from pathlib import Path

import numpy as np


BASE = Path("/home2/zhaoyx/scenario-dreamer/experiments/riskweaver_profile_paired_pilot_20260923_r1")
GROUPS = ("profile_off", "profile_on")


def load_group(name):
    path = BASE / name / "movies" / "multi_scenario_ability_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = {int(row["scenario_index"]): row for row in payload["episodes"]}
    return path, payload["aggregate"], episodes


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def summarize(episodes):
    rows = list(episodes.values())
    valid_d = [float(row["scenario_danger_score"]) for row in rows
               if row.get("danger_valid") and finite_or_none(row.get("scenario_danger_score")) is not None]
    return {
        "scenario_count": len(rows),
        "valid_d_count": len(valid_d),
        "valid_d_coverage": len(valid_d) / len(rows) if rows else None,
        "mean_valid_d": float(np.mean(valid_d)) if valid_d else None,
        "attack_plan_count": int(sum(row.get("attack_plan_count", 0) for row in rows)),
        "attack_executed_frames": int(sum(row.get("attack_executed_frames", 0) for row in rows)),
        "scenes_with_attack": int(sum(row.get("attack_executed_frames", 0) > 0 for row in rows)),
        "generation_failure_count": int(sum(bool(row.get("generation_failure")) for row in rows)),
        "planner_service_failure_count": int(sum(row.get("planner_service_failure_count", 0) for row in rows)),
        "background_collision_frames": int(sum(row.get("feasibility", {}).get("background_collision_frames", 0) for row in rows)),
        "background_static_collision_frames": int(sum(row.get("feasibility", {}).get("background_static_collision_frames", 0) for row in rows)),
        "mean_progress": float(np.mean([row.get("progress", 0.0) for row in rows])) if rows else None,
    }


def paired_bootstrap(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "bootstrap95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "pair_count": int(len(values)),
    }


def main():
    loaded = {name: load_group(name) for name in GROUPS}
    episodes = {name: loaded[name][2] for name in GROUPS}
    shared = sorted(set(episodes["profile_off"]) & set(episodes["profile_on"]))
    paired_rows = []
    danger_differences = []
    attack_frame_differences = []
    for index in shared:
        off = episodes["profile_off"][index]
        on = episodes["profile_on"][index]
        off_d = finite_or_none(off.get("scenario_danger_score")) if off.get("danger_valid") else None
        on_d = finite_or_none(on.get("scenario_danger_score")) if on.get("danger_valid") else None
        if off_d is not None and on_d is not None:
            danger_differences.append(on_d - off_d)
        attack_frame_differences.append(on.get("attack_executed_frames", 0) - off.get("attack_executed_frames", 0))
        paired_rows.append({
            "scenario_index": index,
            "scenario_source": on.get("scenario_source"),
            "off_d": off_d,
            "on_d": on_d,
            "off_attack_frames": off.get("attack_executed_frames", 0),
            "on_attack_frames": on.get("attack_executed_frames", 0),
            "off_failure": off.get("generation_failure"),
            "on_failure": on.get("generation_failure"),
        })

    report = {
        "groups": {name: summarize(episodes[name]) for name in GROUPS},
        "paired": {
            "shared_scenarios": len(shared),
            "danger_on_minus_off": paired_bootstrap(danger_differences),
            "attack_frames_on_minus_off": paired_bootstrap(attack_frame_differences),
            "rows": paired_rows,
        },
        "acceptance": {
            "service_failures_zero": all(summarize(episodes[name])["planner_service_failure_count"] == 0 for name in GROUPS),
            "at_least_three_attack_scenes_each": all(summarize(episodes[name])["scenes_with_attack"] >= 3 for name in GROUPS),
            "background_collisions_zero": all(
                summarize(episodes[name])["background_collision_frames"] == 0
                and summarize(episodes[name])["background_static_collision_frames"] == 0
                for name in GROUPS
            ),
        },
    }
    report["acceptance"]["passed"] = all(report["acceptance"].values())
    (BASE / "paired_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (BASE / "paired_scenarios.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_rows[0].keys())
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
