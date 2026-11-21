"""从只读 r2 派生三秒验收，按后端冻结源码、预检与启动。"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_DIR = ROOT / "work/skills_acceptance_20261109"
ARM = "skills_joint_dual"
MODE_BACKEND = {"diffusion": "diffusion", "cpu_interpolation": "interpolation"}
SOURCES = ("anchor_execution.py", "avoidance_witness.py", "run_anchor_worker.py", "launch_anchor.py")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def derive(protocol, command, mode, frozen, experiment, witness_accel=None):
    """保留 r2 的输入和预算，只替换执行入口及其必要参数。"""
    if len(command) < 2 or Path(command[1]).name != "run_idm_heading.py":
        raise ValueError("unexpected_predecessor_worker")
    backend = MODE_BACKEND[mode]
    worker = [str(HERE / "run_anchor_worker.py"), "--anchor-backend", backend]
    cpu = mode == "cpu_interpolation"
    if cpu:
        worker += ["--interpolation-cpu"]
    if witness_accel is not None:
        if not 2. < witness_accel <= 6.:
            raise ValueError("invalid_explicit_witness_acceleration")
        worker += ["--witness-max-accel-mps2", str(witness_accel)]
    command = [command[0], *worker, *command[2:]]
    updates = {"--manifest": str(frozen / "manifest.json"), "--output-dir": str(experiment / ARM)}
    if cpu:
        updates["--device"] = "cpu"
    for flag, value in updates.items():
        if command.count(flag) != 1:
            raise ValueError(f"unexpected_predecessor_argument:{flag}")
        command[command.index(flag) + 1] = value
    return dict(protocol, version="official80_full_anchor_window_v1", experiment_root=str(experiment),
                source_protocol=str(frozen), predecessor_protocol=protocol["source_protocol"],
                commands={ARM: command}, anchor_backend=backend, anchor_times_s=[0, 1, 2, 3],
                avoidance_witness_max_accel_mps2=witness_accel,
                mode=mode, evaluation_scope="skills_interpolation_acceptance" if cpu else "diffusion_acceptance")


def validate_name(name):
    if not re.fullmatch(r"skills_anchor_(diffusion|interp)20_[0-9]{8}_r[1-9][0-9]*", name):
        raise ValueError("invalid_new_experiment_name")


def derive_current(mode, name, witness_accel):
    """复用 r2 的原始校验器，并冻结新增工作器文件。"""
    from launch_single import check
    from prepare_single import digest

    validate_name(name)
    marker = "_interp20_" if mode == "cpu_interpolation" else "_diffusion20_"
    if marker not in name:
        raise ValueError("backend_experiment_name_mismatch")
    base = BASE_DIR / "protocol_r2"
    protocol, command = check(base)
    frozen = HERE / name
    experiment = ROOT / "experiments" / name
    expected = derive(protocol, command, mode, frozen, experiment, witness_accel)
    hashes = load(base / "source_sha256.json")
    for filename in SOURCES:
        path = HERE / filename
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    expected["source_count"] = len(hashes)
    return base, frozen, experiment, expected, hashes


def runtime_env(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=".:safe-sim:safe-sim/trajdata/src",
                PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                LLM_MODEL_NAME="deepseek-flash", LLM_MODEL_NAMES="deepseek-flash")


def checked_protocol(mode, name, witness_accel=None, prepare=False):
    from prepare_single import digest

    base, frozen, experiment, expected, hashes = derive_current(mode, name, witness_accel)
    if prepare:
        if experiment.exists():
            raise FileExistsError(experiment)
        frozen.mkdir()
        shutil.copyfile(base / "manifest.json", frozen / "manifest.json")
        write(frozen / "source_sha256.json", hashes)
        for relative in hashes:
            destination = frozen / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        # 成功运行准备入口后才写协议。失败目录和日志留存。
        with (frozen / "prepare_only.log").open("x", encoding="utf-8") as log:
            subprocess.run(expected["commands"][ARM] + ["--prepare-only"], cwd=ROOT,
                           env=runtime_env(""), stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, timeout=120, check=True)
        write(frozen / "protocol.json", expected)
    if (load(frozen / "protocol.json") != expected
            or load(frozen / "source_sha256.json") != hashes
            or digest(frozen / "manifest.json") != expected["manifest_sha256"]
            or any(digest(frozen / "source" / name) != value for name, value in hashes.items())):
        raise ValueError("frozen_anchor_protocol_mismatch")
    return frozen, experiment, expected


def eligible_gpus(free_mib):
    result = []
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.free",
                                    "--format=csv,noheader,nounits"], text=True, timeout=20)
    for row in rows.splitlines():
        index, name, free = [value.strip() for value in row.split(",")]
        if "V100S" in name and int(free) >= free_mib:
            result.append((int(free), index))
    return result


def execute(mode, name, frozen, experiment, protocol, gpu):
    """新实验独立落盘并记录终态，任何失败原件保留。"""
    pidfile = experiment.with_suffix(".pid")
    if pidfile.exists():
        raise FileExistsError(pidfile)
    experiment.mkdir()
    for file in ("protocol.json", "source_sha256.json", "manifest.json"):
        shutil.copyfile(frozen / file, experiment / file)
    shutil.copytree(frozen / "source", experiment / "source_snapshot")
    with pidfile.open("x", encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    code = None
    receipt = dict(verified=False, gpu=gpu or None, mode=mode, source_count=protocol["source_count"])
    try:
        with (experiment / f"{ARM}.log").open("x", encoding="utf-8") as log:
            code = subprocess.call(["bash", "scripts/run_with_llm_provider.sh", "deepseek",
                                    *protocol["commands"][ARM]], cwd=ROOT, env=runtime_env(gpu),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        checked_protocol(mode, name, protocol["avoidance_witness_max_accel_mps2"])
        receipt["verified"] = True
    except BaseException as error:
        receipt["error_type"] = type(error).__name__
        raise
    finally:
        write(experiment / "runtime_source_integrity.json", receipt)
        write(experiment / "launcher_result.json",
              dict(status="completed" if code == 0 and receipt["verified"] else "failed",
                   groups={ARM: {"exit_code": code}}, planned_attempts=20, finished_at=time.time()))
    if code:
        raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODE_BACKEND), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--witness-max-accel-mps2", type=float)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(BASE_DIR))
    frozen, experiment, protocol = checked_protocol(args.mode, args.name, args.witness_max_accel_mps2,
                                                    prepare=args.prepare)
    if args.prepare:
        print(json.dumps(dict(frozen=str(frozen), source_count=protocol["source_count"], prepare_only=True)))
        return
    cpu = args.mode == "cpu_interpolation"
    available = [] if cpu else eligible_gpus(protocol["admission_free_mib"])
    disk_ok = shutil.disk_usage(ROOT).free >= 5.5 * 2**30
    ready = (cpu or bool(available)) and disk_ok and not experiment.exists() and not experiment.with_suffix(".pid").exists()
    print(json.dumps(dict(ready=ready, mode=args.mode, eligible_gpus=available, disk_ok=disk_ok)), flush=True)
    if args.launch:
        if not ready:
            raise RuntimeError("anchor_launch_preconditions_not_met")
        execute(args.mode, args.name, frozen, experiment, protocol, "" if cpu else max(available)[1])


if __name__ == "__main__":
    main()
