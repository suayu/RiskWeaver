"""从已冻结 r1 派生 r2。保留原清单和请求预算。"""
import json
import os
import shutil
import subprocess

from launch_single import check
from prepare_single import HERE, ROOT, digest

BASE = HERE/"protocol_r1"
FROZEN = HERE/"protocol_r2"
EXPERIMENT = ROOT/"experiments/skills_official80_acceptance_20261109_r2"


def main():
    protocol, command = check(BASE)
    hashes = json.loads((BASE/"source_sha256.json").read_text())
    for name in ("run_idm_heading.py","prepare_heading_r2.py","test_idm_heading.py"):
        path = HERE/name
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    command = list(command)
    command[1:2] = [str(HERE/"run_idm_heading.py"),"--idm-stationary-heading","on"]
    command[command.index("--manifest")+1] = str(FROZEN/"manifest.json")
    command[command.index("--output-dir")+1] = str(EXPERIMENT/"skills_joint_dual")
    protocol.update(version="official80_rear_window_stationary_heading_skills_only_v2",
        experiment_root=str(EXPERIMENT),source_protocol=str(FROZEN),source_count=len(hashes),
        commands={"skills_joint_dual":command},idm_stationary_heading="hold_previous_when_displacement_zero",
        predecessor_protocol=str(BASE))
    FROZEN.mkdir(exist_ok=False)
    shutil.copyfile(BASE/"manifest.json",FROZEN/"manifest.json")
    (FROZEN/"source_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    for name in hashes:
        target = FROZEN/"source"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    with (FROZEN/"prepare_only.log").open("x",encoding="utf-8") as log:
        subprocess.run(command+["--prepare-only"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ,PYTHONPATH=".:safe-sim:safe-sim/trajdata/src",CUDA_VISIBLE_DEVICES=""),
            check=True,timeout=90)
    (FROZEN/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    check(FROZEN)
    print(json.dumps(dict(protocol=str(FROZEN),sources=len(hashes),actual_requests=0,actual_scenes=0)))


if __name__ == "__main__":
    main()
