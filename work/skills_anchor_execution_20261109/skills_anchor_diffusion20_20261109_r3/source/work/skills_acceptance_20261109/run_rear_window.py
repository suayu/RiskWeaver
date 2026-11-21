"""显式启用后车活动窗事实，再委派给原 Flash 工作进程。"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

from rear_window_evidence import install

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rear-window-evidence", choices=("off", "on"), default="off")
    args, rest = parser.parse_known_args()
    base = ROOT / "work/deepseek_flash_20261106/run_flash_worker_v2.py"
    sys.argv = [str(base), *rest]
    sys.path.insert(0, str(base.parent))
    if args.rear_window_evidence == "off":
        return runpy.run_path(str(base), run_name="__main__")
    if any(flag not in rest or rest[rest.index(flag)+1] != "on"
           for flag in ("--skills", "--incremental-skills")):
        raise ValueError("rear_window_evidence_requires_skills")
    if "--prepare-only" not in rest:
        out = Path(rest[rest.index("--output-dir")+1])
        frozen = json.loads((out.parent/"source_sha256.json").read_text())
        for path in (Path(__file__).resolve(), HERE/"rear_window_evidence.py"):
            if frozen.get(path.relative_to(ROOT).as_posix()) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("rear_window_evidence_not_frozen")
    sys.path[:0] = [str(ROOT), str(ROOT/"work/skills_incremental_evidence_20261025")]
    import incremental_planner
    restore = install(incremental_planner)
    try:
        return runpy.run_path(str(base), run_name="__main__")
    finally:
        restore()


if __name__ == "__main__":
    main()
