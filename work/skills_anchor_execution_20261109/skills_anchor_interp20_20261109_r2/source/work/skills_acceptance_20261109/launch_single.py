"""单组工作器。默认只预检，显式执行才调用现有 Flash 工作进程。"""
import argparse
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

from prepare_single import ROOT, digest


def check(directory):
    protocol = json.loads((directory/"protocol.json").read_text())
    manifest_file = directory/"manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    entries = manifest["provenance"]
    hashes = json.loads((directory/"source_sha256.json").read_text())
    if protocol["arms"] != ["skills_joint_dual"] or protocol["models"] != ["deepseek-flash"]:
        raise ValueError("unexpected_arm_or_provider")
    if (len(entries) != 20 or len({e["sha256"] for e in entries}) != 20
            or sum(e["origin"]=="upstream_200m" for e in entries) != 16
            or digest(manifest_file) != protocol["manifest_sha256"]
            or [e["index"] for e in entries] != protocol["planned_scenes"]
            or [e["path"] for e in entries] != manifest["scenario_files"]
            or [e["sha256"] for e in entries] != manifest["scenario_sha256"]):
        raise ValueError("frozen_manifest_contract")
    for e in entries:
        if digest(Path(e["path"])) != e["sha256"]:
            raise ValueError(f"scene_changed:{e['index']}")
    if len(hashes) != protocol["source_count"]:
        raise ValueError("source_count_changed")
    for name, expected in hashes.items():
        if digest(ROOT/name) != expected or digest(directory/"source"/name) != expected:
            raise ValueError(f"source_drift:{name}")
    command = protocol["commands"]["skills_joint_dual"]
    required = {"--skills":"on", "--rear-window-evidence":"on", "--model":"deepseek-flash",
                "--models":"deepseek-flash", "--steps":"400", "--transport-limit":"4",
                "--video-export":"integer", "--execution-area-policy":"positive_area",
                "--manifest":str(manifest_file),
                "--output-dir":str(Path(protocol["experiment_root"])/"skills_joint_dual")}
    if any(command.count(flag) != 1 or command[command.index(flag)+1] != value
           for flag,value in required.items()):
        raise ValueError("runtime_command_contract")
    return protocol, command


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--launch", action="store_true")
    args = p.parse_args()
    protocol, command = check(args.protocol.resolve())
    output = subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.free",
                                      "--format=csv,noheader,nounits"], text=True, timeout=20)
    available = []
    for row in output.splitlines():
        index, name, free = [x.strip() for x in row.split(",")]
        if "V100S" in name and int(free) >= protocol["admission_free_mib"]:
            available.append((int(free),index))
    disk_ok = shutil.disk_usage(ROOT).free >= 5.5 * 2**30
    root = Path(protocol["experiment_root"])
    ready = bool(available) and disk_ok and not root.exists()
    print(json.dumps(dict(source_verified=True, manifest_verified=True, eligible_gpus=available,
                          disk_ok=disk_ok, ready=ready, check_only=not args.launch)), flush=True)
    if not args.launch:
        return
    if not ready:
        raise RuntimeError("launch_preconditions_not_met")
    root.mkdir(exist_ok=False)
    for name in ("protocol.json","source_sha256.json","manifest.json"):
        shutil.copyfile(args.protocol/name,root/name)
    shutil.copytree(args.protocol/"source",root/"source_snapshot")
    (root.with_suffix(".pid")).write_text(str(os.getpid()), encoding="ascii")
    gpu = max(available)[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=".:safe-sim:safe-sim/trajdata/src",
               PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", LLM_MODEL_NAME="deepseek-flash", LLM_MODEL_NAMES="deepseek-flash")
    receipt = {"verified":True, "gpu":gpu, "source_count":protocol["source_count"]}
    code = None
    try:
        with (root/"skills_joint_dual.log").open("x") as log:
            code = subprocess.call(["bash","scripts/run_with_llm_provider.sh","deepseek",*command],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        check(args.protocol.resolve())
    except Exception as error:
        receipt.update(verified=False, error_type=type(error).__name__)
        raise
    finally:
        (root/"runtime_source_integrity.json").write_text(json.dumps(receipt), encoding="utf-8")
        terminal = dict(status="completed" if code == 0 and receipt["verified"] else "failed",
            groups={"skills_joint_dual":{"exit_code":code}}, planned_attempts=20, finished_at=time.time())
        (root/"launcher_result.json").write_text(json.dumps(terminal), encoding="utf-8")
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
