"""汇总 GPUDrive RL 画像开关配对校准，区分全场风险与攻击条件风险。"""

import csv
import json
from pathlib import Path

import numpy as np


BASE = Path(
    "/home2/zhaoyx/scenario-dreamer/experiments/"
    "riskweaver_rl_profile_paired_calibration_20260923"
)
GROUPS = ("profile_off", "profile_on")


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def load_group(name):
    result_path = BASE / name / "movies" / "multi_scenario_ability_results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    episodes = {int(row["scenario_index"]): row for row in payload["episodes"]}
    returns = []
    for path in sorted((BASE / name / "movies").rglob("strategy_returns.jsonl")):
        returns.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return episodes, returns


def summarize(episodes, returns):
    rows = list(episodes.values())
    valid_scene_d = [
        float(row["scenario_danger_score"])
        for row in rows
        if row.get("danger_valid")
        and finite_or_none(row.get("scenario_danger_score")) is not None
    ]
    executed_returns = [
        row for row in returns if int(row.get("actual_attack_execution_frames", 0)) > 0
    ]
    valid_attack_returns = [
        row
        for row in executed_returns
        if finite_or_none(row.get("D")) is not None
        and int(row.get("valid_observation_frames", 0)) > 0
    ]
    attack_weight = sum(int(row["valid_observation_frames"]) for row in valid_attack_returns)
    attack_d = (
        sum(float(row["D"]) * int(row["valid_observation_frames"]) for row in valid_attack_returns)
        / attack_weight
        if attack_weight
        else None
    )
    # 能力分依赖有效危险度；无效危险度对应的能力分保持为空，不参与均值。
    valid_ability_scores = [
        finite_or_none(row.get("autonomous_driving_ability_score")) for row in rows
    ]
    valid_ability_scores = [value for value in valid_ability_scores if value is not None]
    return {
        "scenario_count": len(rows),
        "valid_scene_d_count": len(valid_scene_d),
        "valid_scene_d_coverage": len(valid_scene_d) / len(rows) if rows else None,
        "mean_valid_scene_d": float(np.mean(valid_scene_d)) if valid_scene_d else None,
        "attack_plan_count": int(sum(row.get("attack_plan_count", 0) for row in rows)),
        "attack_executed_frames": int(sum(row.get("attack_executed_frames", 0) for row in rows)),
        "scenes_with_attack": int(sum(row.get("attack_executed_frames", 0) > 0 for row in rows)),
        "executed_strategy_return_count": len(executed_returns),
        "valid_attack_return_count": len(valid_attack_returns),
        "attack_valid_observation_frames": attack_weight,
        "attack_conditioned_d": attack_d,
        "generation_failure_count": int(sum(bool(row.get("generation_failure")) for row in rows)),
        "planner_service_failure_count": int(
            sum(row.get("planner_service_failure_count", 0) for row in rows)
        ),
        "background_collision_frames": int(
            sum(row.get("feasibility", {}).get("background_collision_frames", 0) for row in rows)
        ),
        "background_static_collision_frames": int(
            sum(
                row.get("feasibility", {}).get("background_static_collision_frames", 0)
                for row in rows
            )
        ),
        "valid_ability_score_count": len(valid_ability_scores),
        "valid_ability_score_coverage": (
            len(valid_ability_scores) / len(rows) if rows else None
        ),
        "mean_valid_ability_score": (
            float(np.mean(valid_ability_scores)) if valid_ability_scores else None
        ),
    }


def paired_bootstrap(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "bootstrap95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "pair_count": int(len(values)),
    }


def main():
    loaded = {name: load_group(name) for name in GROUPS}
    episodes = {name: loaded[name][0] for name in GROUPS}
    summaries = {name: summarize(*loaded[name]) for name in GROUPS}
    shared = sorted(set(episodes["profile_off"]) & set(episodes["profile_on"]))
    paired_rows = []
    scene_d_differences = []
    attack_frame_differences = []
    for index in shared:
        off = episodes["profile_off"][index]
        on = episodes["profile_on"][index]
        off_d = finite_or_none(off.get("scenario_danger_score")) if off.get("danger_valid") else None
        on_d = finite_or_none(on.get("scenario_danger_score")) if on.get("danger_valid") else None
        if off_d is not None and on_d is not None:
            scene_d_differences.append(on_d - off_d)
        off_frames = int(off.get("attack_executed_frames", 0))
        on_frames = int(on.get("attack_executed_frames", 0))
        attack_frame_differences.append(on_frames - off_frames)
        paired_rows.append(
            {
                "scenario_index": index,
                "scenario_source": on.get("scenario_source"),
                "off_scene_d": off_d,
                "on_scene_d": on_d,
                "off_attack_frames": off_frames,
                "on_attack_frames": on_frames,
                "off_failure": off.get("generation_failure"),
                "on_failure": on.get("generation_failure"),
            }
        )

    acceptance = {
        "service_failures_zero": all(
            summaries[name]["planner_service_failure_count"] == 0 for name in GROUPS
        ),
        "at_least_one_attack_scene_each": all(
            summaries[name]["scenes_with_attack"] >= 1 for name in GROUPS
        ),
        "background_collisions_zero": all(
            summaries[name]["background_collision_frames"] == 0
            and summaries[name]["background_static_collision_frames"] == 0
            for name in GROUPS
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    report = {
        "scope": "three-scene GPUDrive RL calibration; not formal paper evidence",
        "groups": summaries,
        "paired": {
            "shared_scenarios": len(shared),
            "scene_d_on_minus_off": paired_bootstrap(scene_d_differences),
            "attack_frames_on_minus_off": paired_bootstrap(attack_frame_differences),
            "rows": paired_rows,
        },
        "acceptance": acceptance,
    }
    (BASE / "paired_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (BASE / "paired_scenarios.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_rows[0].keys())
        writer.writeheader()
        writer.writerows(paired_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
