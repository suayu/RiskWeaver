"""汇总 RiskWeaver 20 场 IDM/RL × 自车画像开关正式实验。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


GROUPS = {
    "idm_profile_off": ("idm", False),
    "idm_profile_on": ("idm", True),
    "rl_profile_off": ("rl", False),
    "rl_profile_on": ("rl", True),
}


def finite(value):
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def bootstrap_mean(values, seed=42, samples=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "bootstrap95": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "pair_count": int(len(values)),
    }


def load_group(base, group):
    result_path = base / group / "movies/multi_scenario_ability_results.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    episodes = {
        Path(row["scenario_source"]).name: row for row in payload["episodes"]
    }
    if len(episodes) != len(payload["episodes"]):
        raise ValueError(f"{group}: 场景来源重复")
    returns = []
    for path in sorted((base / group / "movies").rglob("strategy_returns.jsonl")):
        returns.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return episodes, returns


def summarize(rows, returns):
    rows = list(rows)
    valid_d = [
        finite(row.get("scenario_danger_score"))
        for row in rows
        if row.get("danger_valid")
    ]
    valid_d = [value for value in valid_d if value is not None]
    ability = [
        finite(row.get("autonomous_driving_ability_score")) for row in rows
    ]
    ability = [value for value in ability if value is not None]
    executed_returns = [
        row for row in returns if int(row.get("actual_attack_execution_frames", 0)) > 0
    ]
    valid_returns = [
        row for row in executed_returns
        if finite(row.get("D")) is not None
        and int(row.get("valid_observation_frames", 0)) > 0
    ]
    valid_frames = sum(int(row["valid_observation_frames"]) for row in valid_returns)
    return {
        "scenario_count": len(rows),
        "attack_plan_count": int(sum(row.get("attack_plan_count", 0) for row in rows)),
        "attack_scene_count": int(sum(int(row.get("attack_executed_frames", 0)) > 0 for row in rows)),
        "attack_scene_rate": float(np.mean([
            int(row.get("attack_executed_frames", 0)) > 0 for row in rows
        ])) if rows else None,
        "attack_executed_frames": int(sum(row.get("attack_executed_frames", 0) for row in rows)),
        "valid_scene_d_count": len(valid_d),
        "valid_scene_d_coverage": len(valid_d) / len(rows) if rows else None,
        "mean_valid_scene_d": float(np.mean(valid_d)) if valid_d else None,
        "attack_conditioned_d": (
            sum(float(row["D"]) * int(row["valid_observation_frames"]) for row in valid_returns)
            / valid_frames if valid_frames else None
        ),
        "valid_attack_return_count": len(valid_returns),
        "attack_valid_observation_frames": int(valid_frames),
        "ego_collision_count": int(sum(bool(row.get("collision")) for row in rows)),
        "completed_count": int(sum(bool(row.get("completed")) for row in rows)),
        "generation_failure_count": int(sum(bool(row.get("generation_failure")) for row in rows)),
        "planner_service_failure_count": int(sum(
            row.get("planner_service_failure_count", 0) for row in rows
        )),
        "background_collision_frames": int(sum(
            row.get("feasibility", {}).get("background_collision_frames", 0)
            for row in rows
        )),
        "background_static_collision_frames": int(sum(
            row.get("feasibility", {}).get("background_static_collision_frames", 0)
            for row in rows
        )),
        "mean_valid_ability_score": float(np.mean(ability)) if ability else None,
        "valid_ability_score_count": len(ability),
    }


def pair_rows(off, on):
    shared = sorted(set(off) & set(on))
    records = []
    differences = {
        "attack_scene_rate": [],
        "attack_frames": [],
        "scene_d": [],
        "ability_score": [],
    }
    for source in shared:
        left, right = off[source], on[source]
        left_attack = int(left.get("attack_executed_frames", 0))
        right_attack = int(right.get("attack_executed_frames", 0))
        left_d = finite(left.get("scenario_danger_score")) if left.get("danger_valid") else None
        right_d = finite(right.get("scenario_danger_score")) if right.get("danger_valid") else None
        left_ability = finite(left.get("autonomous_driving_ability_score"))
        right_ability = finite(right.get("autonomous_driving_ability_score"))
        differences["attack_scene_rate"].append(float(right_attack > 0) - float(left_attack > 0))
        differences["attack_frames"].append(right_attack - left_attack)
        if left_d is not None and right_d is not None:
            differences["scene_d"].append(right_d - left_d)
        if left_ability is not None and right_ability is not None:
            differences["ability_score"].append(right_ability - left_ability)
        records.append({
            "scenario_source": source,
            "off_attack_frames": left_attack,
            "on_attack_frames": right_attack,
            "off_scene_d": left_d,
            "on_scene_d": right_d,
            "off_ability_score": left_ability,
            "on_ability_score": right_ability,
            "off_collision": bool(left.get("collision")),
            "on_collision": bool(right.get("collision")),
            "off_generation_failure": left.get("generation_failure"),
            "on_generation_failure": right.get("generation_failure"),
        })
    return records, {
        name: bootstrap_mean(values) for name, values in differences.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.base / "formal_summary"
    output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads((args.base / "scenario_manifest.json").read_text(encoding="utf-8"))
    split_by_name = {
        f"{item['scenario_id']}.pkl": item["split"]
        for item in manifest["scenario_metadata"]
    }
    loaded = {group: load_group(args.base, group) for group in GROUPS}
    expected = set(Path(path).name for path in manifest["scenario_files"])
    for group, (episodes, _) in loaded.items():
        if set(episodes) != expected:
            raise ValueError(
                f"{group}: 实际场景与冻结清单不一致，missing={sorted(expected-set(episodes))}, "
                f"extra={sorted(set(episodes)-expected)}"
            )

    summary = {}
    for group, (episodes, returns) in loaded.items():
        summary[group] = {
            "overall": summarize(episodes.values(), returns),
            "test": summarize(
                [row for name, row in episodes.items() if split_by_name[name] == "test"],
                [row for row in returns if split_by_name.get(row.get("scene_id")) == "test"],
            ),
            "validation": summarize(
                [row for name, row in episodes.items() if split_by_name[name] == "validation"],
                [row for row in returns if split_by_name.get(row.get("scene_id")) == "validation"],
            ),
        }

    comparisons = {}
    all_rows = []
    for policy in ("idm", "rl"):
        off = loaded[f"{policy}_profile_off"][0]
        on = loaded[f"{policy}_profile_on"][0]
        rows, paired = pair_rows(off, on)
        for row in rows:
            row["policy"] = policy
            row["split"] = split_by_name[row["scenario_source"]]
        all_rows.extend(rows)
        comparisons[policy] = {"paired": paired, "rows": rows}

    integrity = {
        "all_groups_have_20_frozen_scenarios": all(
            data["overall"]["scenario_count"] == 20 for data in summary.values()
        ),
        "planner_service_failures_zero": all(
            data["overall"]["planner_service_failure_count"] == 0
            for data in summary.values()
        ),
        "background_collisions_zero": all(
            data["overall"]["background_collision_frames"] == 0
            and data["overall"]["background_static_collision_frames"] == 0
            for data in summary.values()
        ),
    }
    integrity["passed"] = all(integrity.values())
    report = {
        "scope": "frozen 20-scene IDM/RL ego-profile formal experiment",
        "groups": summary,
        "comparisons": comparisons,
        "integrity": integrity,
    }
    (output / "formal_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (output / "formal_paired_scenarios.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

