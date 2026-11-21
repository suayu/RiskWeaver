"""终态重放全部源帧，使用同一真实联合未来对比两种可达面积。"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
from omegaconf import OmegaConf

ROOT = Path("/home2/zhaoyx/scenario-dreamer")
sys.path.insert(0, str(ROOT))
from policies.risk_metrics import RouteReachability
from policies.commonroad_reachability import CommonRoadReachability
from policies.traffic_types import JointTrajectory
from experiments.comparison.ctrl_sim_forecast import validate_forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arm = args.experiment / "skills_joint_dual"
    assert json.loads((arm / "exit_status.json").read_text())["exit_code"] == 0
    args.output.mkdir(exist_ok=False)
    manifest = json.loads((args.experiment / "manifest.json").read_text())
    cfg = OmegaConf.load(arm / "resolved_config.yaml").sim
    grid = RouteReachability(cfg.evaluation.reachability)
    settings = OmegaConf.to_container(cfg.evaluation.reachability, resolve=True)
    settings["commonroad_inflation_mode"] = 2
    circle = CommonRoadReachability(settings)
    summaries = []
    try:
        for index, scene in enumerate(manifest["original_scenario_indices"]):
            payload = Path(manifest["scenario_files"][index]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == manifest["scenario_sha256"][index]
            geometry = pickle.loads(payload)
            trace = arm / f"movies/scenario_{scene:03d}/execution_trace.jsonl"
            raw = trace.read_bytes()
            records = list(map(json.loads, raw.splitlines()))
            states = [r for r in records if r["kind"] == "state"]
            predictions = {}
            for row in records:
                if row["kind"] == "execution_prediction":
                    predictions[row["step"]] = row["joint"]
            measurements = []
            with (args.output / f"scene_{scene:03d}.jsonl").open("x") as output:
                for state in states:
                    step = state["step"]
                    item = dict(step=step, route_grid=None, commonroad_circle=None)
                    try:
                        j = predictions.get(step)
                        if j is None:
                            raise ValueError("no_native_future_at_terminal_or_rejected_source")
                        ids = np.asarray(j["agent_ids"], int)
                        horizon = len(j["valid_mask"][0]) if len(ids) else int(cfg.traffic_model.prediction_horizon)
                        joint = JointTrajectory(step, ids,
                            np.asarray(j["positions_global"]).reshape(len(ids), horizon, 2),
                            np.asarray(j["yaws_global"]).reshape(len(ids), horizon, 1),
                            np.asarray(j["velocities_global"]).reshape(len(ids), horizon, 2),
                            np.asarray(j["valid_mask"]).reshape(len(ids), horizon), j["metadata"])
                        env = NS(scenario_dict=geometry, ego_state=np.asarray(state["ego"]),
                                 data_dict={"agent": [np.asarray(state["agents"])]},
                                 agent_active=np.asarray(state["active"], bool), dt=float(cfg.dt),
                                 current_step=step, pending_joint_trajectory=joint)
                        validate_forecast(joint, env, grid.horizon)
                        for name, solver in (("route_grid", grid), ("commonroad_circle", circle)):
                            start = time.monotonic()
                            try:
                                item[name] = dict(status="valid", **solver.compute(env, True, state["static_obstacles"]))
                            except Exception as error:
                                item[name] = dict(status="invalid", area_m2=None, reason=repr(error))
                            item[name]["evaluation_wall_seconds"] = time.monotonic() - start
                    except Exception as error:
                        item["input_invalid_reason"] = repr(error)
                    output.write(json.dumps(item, allow_nan=False) + "\n")
                    output.flush()
                    measurements.append(item)
                    (args.output / "progress.json").write_text(json.dumps(dict(scene=scene, source_frames=len(measurements))))
            pairs = [m for m in measurements if all(m[k] and m[k]["status"] == "valid" for k in ("route_grid", "commonroad_circle"))]
            summaries.append(dict(scene=scene, all_recorded_states=len(states), paired_valid_sources=len(pairs),
                                  trace_sha256=hashlib.sha256(raw).hexdigest(),
                                  grid_mean_area_m2=float(np.mean([m["route_grid"]["area_m2"] for m in pairs])) if pairs else None,
                                  circle_mean_area_m2=float(np.mean([m["commonroad_circle"]["area_m2"] for m in pairs])) if pairs else None))
            (args.output / "summary.partial.json").write_text(json.dumps(summaries, indent=2))
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
        (args.output / "exit_status.json").write_text(json.dumps(dict(exit_code=0, scenes=len(summaries))))
    finally:
        circle.close()


if __name__ == "__main__":
    main()
