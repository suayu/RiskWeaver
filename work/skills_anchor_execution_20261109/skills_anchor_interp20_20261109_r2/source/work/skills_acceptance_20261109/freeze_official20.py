"""按上游 blob 校验并冻结一组 Skills 专用清单。只读输入目录。"""
import hashlib
import json
from pathlib import Path
import random
import subprocess

ROOT = Path.cwd()
POOL = Path("experiments/scenario_dreamer_ldm_expand250_20260928_r1/riskweaver_frozen_250_scenes.json")
PREVIOUS = Path("experiments/skills_deepseek_flash_paired20_20261106_r3/manifest.json")
OFFICIAL = Path("metadata/simulation_environment_datasets/scenario_dreamer_waymo_200m_pickles")
SEED = 20261109


def load(path):
    return json.loads(path.read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    pool, previous = load(POOL), load(PREVIOUS)
    seen = set(previous["scenario_sha256"])
    ref = git("rev-parse", "upstream/main")
    official_entries = {}
    for line in git("ls-tree", "-r", ref, str(OFFICIAL)).splitlines():
        object_info, relative = line.split("\t", 1)
        if not relative.endswith(".pkl"):
            continue
        _, kind, blob = object_info.split()
        if kind != "blob":
            raise ValueError(f"not a Git blob: {relative}")
        raw = (ROOT / relative).read_bytes()
        git_hash = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if git_hash != blob:
            raise ValueError(f"not identical to upstream: {relative}")
        official_entries[str(ROOT / relative)] = (blob, digest(raw))
    if len(official_entries) != 75:
        raise ValueError("unexpected upstream official scene count")

    official, generated = [], {}
    for index, (path, sha) in enumerate(zip(pool["scenario_files"], pool["scenario_sha256"], strict=True)):
        if sha in seen:
            continue
        entry = dict(index=index, path=path, sha256=sha)
        if path in official_entries:
            blob, official_sha = official_entries[path]
            if sha != official_sha:
                raise ValueError("official hash conflicts with pool")
            official.append(dict(entry, origin="upstream_200m", upstream_blob=blob))
        elif Path(path).parent.name == "postprocessed_200m_pickles":
            group = Path(path).parent.parent.name
            generated.setdefault(group, []).append(dict(entry, origin=group))
        else:
            raise ValueError(f"unrecognized scenario provenance: {path}")

    rng = random.Random(SEED)
    selection = rng.sample(official, 16)
    if len(generated) != 2:
        raise ValueError("expected two frozen generated sources")
    for name in sorted(generated):
        selection += rng.sample(generated[name], 2)
    rng.shuffle(selection)
    assert len({e["sha256"] for e in selection}) == 20
    for e in selection:
        if digest(Path(e["path"]).read_bytes()) != e["sha256"]:
            raise ValueError(f"scene changed: {e['path']}")

    # 字段兼容旧单组运行器，来源证据附在独立记录中。
    print(json.dumps(dict(
        name="skills_only_official80_frozen20_20261109_r1", selection_seed=SEED,
        seed=42, simulation_steps=400, scenario_count=20, comparison_scene_count=20,
        selection_rule="exclude previous paired20 SHA256; random 16 upstream + 2 per frozen generated source",
        statistical_unit="frozen_unique_scenario",
        upstream_ref=ref, upstream_inventory_verified=75, official_planned=16,
        previous_manifest_sha256=digest(PREVIOUS.read_bytes()),
        pool_manifest_sha256=digest(POOL.read_bytes()),
        scenario_files=[e["path"] for e in selection],
        scenario_sha256=[e["sha256"] for e in selection],
        original_scenario_indices=[e["index"] for e in selection],
        provenance=selection,
        acceptance=dict(group="skills_only", required_effective=18, required_ego_collisions=11,
                        minimum_official_videos=16, never_pad_video=True),
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
