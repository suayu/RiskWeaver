"""仅在本工作进程启用 IDM 停车航向修复，随后使用原 Skills 执行链。"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = """        lane_angle = np.arctan2(pos_delta[1], pos_delta[0])
        ego_next_theta = (lane_angle + PI) % (2 * PI) - PI  # Corrected angle"""
NEW = """        if np.linalg.norm(pos_delta) <= 1e-8:
            # 零位移没有方向。延用当前航向，避免停车时车头瞬时归零。
            ego_next_theta = float(agent_states[ego_id, 4])
        else:
            lane_angle = np.arctan2(pos_delta[1], pos_delta[0])
            ego_next_theta = (lane_angle + PI) % (2 * PI) - PI"""


def install(idm_class):
    original = idm_class._get_next_states
    source = textwrap.dedent(inspect.getsource(original))
    if source.count(OLD) != 1:
        raise ValueError("IDM_heading_source_contract_changed")
    namespace = {}
    exec(compile(source.replace(OLD, NEW), inspect.getfile(original), "exec"),
         original.__globals__, namespace)
    idm_class._get_next_states = namespace[original.__name__]

    def restore():
        idm_class._get_next_states = original
    return restore


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--idm-stationary-heading", choices=("off","on"), default="off")
    options, args = parser.parse_known_args()
    delegate = HERE/"run_rear_window.py"
    sys.argv = [str(delegate), *args]
    if options.idm_stationary_heading == "off":
        return runpy.run_path(str(delegate), run_name="__main__")
    if "--prepare-only" not in args:
        out = Path(args[args.index("--output-dir")+1])
        frozen = json.loads((out.parent/"source_sha256.json").read_text())
        for path in (Path(__file__).resolve(), ROOT/"policies/idm_policy.py"):
            if hashlib.sha256(path.read_bytes()).hexdigest() != frozen.get(path.relative_to(ROOT).as_posix()):
                raise ValueError("IDM_heading_source_not_frozen")
    sys.path.insert(0,str(ROOT))
    from policies.idm_policy import IDMPolicy
    restore = install(IDMPolicy)
    try:
        return runpy.run_path(str(delegate), run_name="__main__")
    finally:
        restore()


if __name__ == "__main__":
    main()
