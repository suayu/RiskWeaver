"""冻结 Skills 单组协议。仅准备和离线检查，不请求模型或执行场景。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "experiments/skills_deepseek_flash_paired20_20261106_r3"
FROZEN = HERE / "protocol_r1"
EXPERIMENT = ROOT / "experiments/skills_official80_acceptance_20261109_r1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8-sig"))
    hashes = json.loads((PREVIOUS / "source_sha256.json").read_text())
    mismatch = [name for name, expected in hashes.items() if digest(ROOT/name) != expected]
    if mismatch:
        raise ValueError(f"previous_runtime_source_drift: {mismatch}")
    for name in ("freeze_official20.py", "rear_window_evidence.py", "run_rear_window.py",
                 "prepare_single.py", "launch_single.py", "audit_single.py"):
        path = HERE / name
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    # 旧快照的技能叶文档有独立完整性要求，同样纳入这次源码冻结。
    manifest_path = ROOT / "skills/riskweaver/manifest.json"
    skills = json.loads(manifest_path.read_text())
    for relative, expected in skills["files"].items():
        file = manifest_path.parent / relative
        if digest(file) != expected:
            raise ValueError(f"skill_manifest_drift: {relative}")
        hashes[file.relative_to(ROOT).as_posix()] = expected

    command = json.loads((PREVIOUS / "protocol.json").read_text())["commands"]["skills_joint_dual"]
    command[1:2] = [str(HERE / "run_rear_window.py"), "--rear-window-evidence", "on"]
    command[command.index("--manifest")+1] = str(FROZEN / "manifest.json")
    command[command.index("--output-dir")+1] = str(EXPERIMENT / "skills_joint_dual")
    FROZEN.mkdir(exist_ok=False)
    shutil.copyfile(HERE / "manifest.json", FROZEN / "manifest.json")
    (FROZEN/"source_sha256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    for name in hashes:
        target = FROZEN/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    protocol = dict(version="official80_rear_window_skills_only_v1", experiment_root=str(EXPERIMENT),
        source_protocol=str(FROZEN), arms=["skills_joint_dual"], commands={"skills_joint_dual":command},
        provider="deepseek", models=["deepseek-flash"], planned_attempts=20, max_requests_per_scene=4,
        planned_scenes=manifest["original_scenario_indices"], manifest_sha256=digest(FROZEN/"manifest.json"),
        source_count=len(hashes), admission_free_mib=18*1024, minimum_effective=18,
        minimum_ego_collisions=11, minimum_official_videos=16, prepare_only=True)
    with (FROZEN/"prepare_only.log").open("x", encoding="utf-8") as log:
        subprocess.run(command+["--prepare-only"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, PYTHONPATH=".:safe-sim:safe-sim/trajdata/src", CUDA_VISIBLE_DEVICES=""),
                       check=True, timeout=90)
    (FROZEN/"protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(dict(protocol=str(FROZEN), frozen_sources=len(hashes), prepared=True,
                          actual_requests=0, actual_scenes=0)))


if __name__ == "__main__":
    main()
