import json

import pytest

from summarize_method_comparison import paired_bootstrap, summarize


def write_result(path, method, episodes):
    path.write_text(json.dumps({"method": method, "episodes": episodes}), encoding="utf-8")


def episode(index, ttc, danger, valid=True):
    return {
        "scenario_index": index,
        "scenario_source": f"/data/{index}.pkl",
        "collision": False,
        "completed": False,
        "off_route": False,
        "progress": 0.2 + index / 10,
        "min_ttc_s": ttc,
        "max_evasive_acceleration_mps2": 1.0 + index,
        "scenario_danger_score": danger,
        "danger_valid": valid,
        "autonomous_driving_ability_score": danger,
        "ability_valid": valid,
        "feasibility": {
            "background_collision_frames": 0,
            "background_static_collision_frames": 0,
        },
    }


def test_summarize_preserves_null_and_pairs_common_scenes(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_result(a, "a", [episode(0, 2.0, 0.4), episode(1, 3.0, None, False)])
    write_result(b, "b", [episode(0, 4.0, None, False), episode(1, 5.0, None, False)])
    report = summarize({"a": a, "b": b}, tmp_path / "out", bootstrap_samples=200)
    assert report["aggregate"]["a"]["danger_valid_count"] == 1
    assert report["aggregate"]["b"]["mean_valid_danger"] is None
    assert report["paired_comparisons"]["a_minus_b"]["danger"]["paired_count"] == 0
    assert report["paired_comparisons"]["a_minus_b"]["min_ttc_s"]["mean_difference"] == -2.0


def test_summarize_rejects_unpaired_scene_sets(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_result(a, "a", [episode(0, 2.0, 0.4)])
    write_result(b, "b", [episode(1, 3.0, 0.5)])
    with pytest.raises(ValueError, match="场景集合不一致"):
        summarize({"a": a, "b": b}, tmp_path / "out", bootstrap_samples=20)
