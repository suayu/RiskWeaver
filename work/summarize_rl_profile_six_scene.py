"""合并两个 GPUDrive RL 三场批次，以场景来源为键生成六场配对汇总。"""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home2/zhaoyx/scenario-dreamer")
GROUP_BASES = {
    "profile_off": (
        ROOT / "experiments/riskweaver_rl_profile_paired_calibration_20260923/profile_off",
        ROOT / "experiments/riskweaver_rl_profile_paired_calibration_extension_20260923/profile_off",
    ),
    "profile_on": (
        ROOT / "experiments/riskweaver_rl_profile_paired_calibration_20260923/profile_on",
        ROOT / "experiments/riskweaver_rl_profile_paired_calibration_extension_20260923/profile_on",
        ROOT / "experiments/riskweaver_rl_profile_on_recovery_8_2_20260923/profile_on",
    ),
}
OUTPUT = ROOT / "experiments/riskweaver_rl_profile_six_scene_summary_20260923"
GROUPS = ("profile_off", "profile_on")


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def load_group(group):
    episodes = {}
    returns = []
    for base in GROUP_BASES[group]:
        # 失败批次可能没有总表；逐场景产物是中断后仍然有效的权威记录。
        for path in sorted((base / "movies").rglob("attempt_result.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            key = Path(row["scenario_source"]).name
            if key in episodes:
                raise ValueError(f"duplicate scenario source across batches: {key}")
            episodes[key] = row
        for path in sorted((base / "movies").rglob("strategy_returns.jsonl")):
            returns.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return episodes, returns


def summarize(episodes, returns):
    rows = list(episodes.values())
    valid_scene_d = [
        finite_or_none(row.get("scenario_danger_score"))
        for row in rows
        if row.get("danger_valid")
    ]
    valid_scene_d = [value for value in valid_scene_d if value is not None]
    executed_returns = [
        row for row in returns if int(row.get("actual_attack_execution_frames", 0)) > 0
    ]
    valid_returns = [
        row
        for row in executed_returns
        if finite_or_none(row.get("D")) is not None
        and int(row.get("valid_observation_frames", 0)) > 0
    ]
    attack_weight = sum(int(row["valid_observation_frames"]) for row in valid_returns)
    ability_scores = [
        finite_or_none(row.get("autonomous_driving_ability_score")) for row in rows
    ]
    ability_scores = [value for value in ability_scores if value is not None]
    return {
        "scenario_count": len(rows),
        "valid_scene_d_count": len(valid_scene_d),
        "valid_scene_d_coverage": len(valid_scene_d) / len(rows) if rows else None,
        "mean_valid_scene_d": float(np.mean(valid_scene_d)) if valid_scene_d else None,
        "attack_plan_count": int(sum(row.get("attack_plan_count", 0) for row in rows)),
        "attack_executed_frames": int(sum(row.get("attack_executed_frames", 0) for row in rows)),
        "scenes_with_attack": int(sum(row.get("attack_executed_frames", 0) > 0 for row in rows)),
        "attack_conditioned_d": (
            sum(float(row["D"]) * int(row["valid_observation_frames"]) for row in valid_returns)
            / attack_weight
            if attack_weight
            else None
        ),
        "attack_valid_observation_frames": attack_weight,
        "valid_attack_return_count": len(valid_returns),
        "ego_collision_count": int(sum(bool(row.get("collision")) for row in rows)),
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
        "valid_ability_score_count": len(ability_scores),
        "mean_valid_ability_score": (
            float(np.mean(ability_scores)) if ability_scores else None
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
    loaded = {group: load_group(group) for group in GROUPS}
    episodes = {group: loaded[group][0] for group in GROUPS}
    summaries = {group: summarize(*loaded[group]) for group in GROUPS}
    shared = sorted(set(episodes["profile_off"]) & set(episodes["profile_on"]))
    rows = []
    scene_d_diff = []
    attack_frame_diff = []
    for source in shared:
        off = episodes["profile_off"][source]
        on = episodes["profile_on"][source]
        off_d = finite_or_none(off.get("scenario_danger_score")) if off.get("danger_valid") else None
        on_d = finite_or_none(on.get("scenario_danger_score")) if on.get("danger_valid") else None
        if off_d is not None and on_d is not None:
            scene_d_diff.append(on_d - off_d)
        off_frames = int(off.get("attack_executed_frames", 0))
        on_frames = int(on.get("attack_executed_frames", 0))
        attack_frame_diff.append(on_frames - off_frames)
        rows.append(
            {
                "scenario_source": source,
                "off_scene_d": off_d,
                "on_scene_d": on_d,
                "off_attack_frames": off_frames,
                "on_attack_frames": on_frames,
                "off_collision": bool(off.get("collision")),
                "on_collision": bool(on.get("collision")),
                "off_failure": off.get("generation_failure"),
                "on_failure": on.get("generation_failure"),
            }
        )
    acceptance = {
        "service_failures_zero": all(
            summaries[group]["planner_service_failure_count"] == 0 for group in GROUPS
        ),
        "at_least_one_attack_scene_each": all(
            summaries[group]["scenes_with_attack"] >= 1 for group in GROUPS
        ),
        "background_collisions_zero": all(
            summaries[group]["background_collision_frames"] == 0
            and summaries[group]["background_static_collision_frames"] == 0
            for group in GROUPS
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    report = {
        "scope": "combined six-scene GPUDrive RL calibration; not formal evidence",
        "source_batches": {group: [str(path) for path in GROUP_BASES[group]] for group in GROUPS},
        "groups": summaries,
        "paired": {
            "shared_scenarios": len(shared),
            "scene_d_on_minus_off": paired_bootstrap(scene_d_diff),
            "attack_frames_on_minus_off": paired_bootstrap(attack_frame_diff),
            "rows": rows,
        },
        "acceptance": acceptance,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "combined_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (OUTPUT / "combined_scenarios.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
