"""七个双空目录初态的原验证器逐项诊断；不改阈值或接受结果。"""
import contextlib
import io
import json
import math
import os
import pickle
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from omegaconf import OmegaConf

from audit import ROOT, initial_environment, save, sha
from policies.ego_profile import CandidateBuilder
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.risk_metrics import RouteReachability
from policies.static_attack_catalog import build_static_combinations


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU_only_required')
    home = Path(__file__).resolve().parent
    result = json.loads((home/'union250_r1/stage_result.json').read_text())
    assert result['status'] == 'completed' and result['scanned'] == 250
    manifest = json.loads((home/'all250_r1/frozen_manifest.json').read_text())
    protected = json.loads((home/'all250_r1/protocol.json').read_text())['source_sha256']
    assert all(sha(ROOT/name) == digest for name,digest in protected.items())
    cfg = OmegaConf.load(ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
    out = home/'gap_forensics7_r1'
    out.mkdir(exist_ok=False)
    rows = []
    for index in result['both_empty_scenes']:
        filename = manifest['scenario_files'][index]
        assert sha(filename) == manifest['scenario_sha256'][index]
        geometry = pickle.loads(Path(filename).read_bytes())
        geometry['_audit_filename'] = Path(filename).name
        env = initial_environment(geometry, cfg)
        planner = LLMAdversarialPlanner(model_name='qwen3.8-2.4t-a95b',
            model_names=['qwen3.8-2.4t-a95b'], provider='dashscope', client=NS())
        planner.use_obstacles = True
        original = planner._validate_obstacle_plan
        calls = []
        def recorded(placements, state):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                returned = original(placements, state)
            edges = []
            for placement in placements:
                obstacles = planner.obstacle_catalog.materialize(placement, new_id=lambda name:'geometry-probe-'+name)
                edge = min(float(o.center_global[1]) - .5*(abs(math.sin(o.yaw))*o.length
                    + abs(math.cos(o.yaw))*o.width) for o in obstacles)
                edges.append(edge)
            calls.append(dict(normalized_placements=placements, minimum_forward_edges_m=edges,
                original_returned_count=len(returned), original_stdout=stream.getvalue()))
            return returned
        planner._validate_obstacle_plan = recorded
        catalog, audit = build_static_combinations(env, planner, CandidateBuilder(cfg.sim.ego_profile))
        route, arc, tangent = RouteReachability._route_geometry(env.scenario_dict['route'])
        ego = env.ego_state
        forward = np.array([math.cos(ego[4]), math.sin(ego[4])])
        start = arc[int(np.argmin(np.linalg.norm(route-ego[:2], axis=1)))]
        samples = []
        for distance in [20.,27.,35.]:
            center, direction = RouteReachability._sample_route(route,arc,tangent,start+distance)
            samples.append(dict(route_arclength_ahead_m=distance,
                current_heading_forward_projection_m=float((center-ego[:2])@forward)))
        row = dict(scene=index, remaining_route_m=float(arc[-1]-start),
            maximum_route_center_current_heading_projection_m=float(np.max((route-ego[:2])@forward)),
            route_samples=samples, original_static_candidates=len(catalog),
            original_validator_calls=len(calls),
            all_placement_edges_below_original_15m=all(all(x<15 for x in c['minimum_forward_edges_m']) for c in calls),
            all_original_calls_rejected=all(c['original_returned_count']==0 for c in calls),
            original_validator_calls_raw=calls, static_audit=audit,
            thresholds_or_safety_changed=False, actual_execution_frames=0, D=None)
        rows.append(row)
        save(out/f'scene{index:03d}.json',row)
    assert all(sha(ROOT/name) == digest for name,digest in protected.items())
    save(out/'stage_result.json',dict(status='completed', scenes=result['both_empty_scenes'],
        all_original_15m_forward_edge_rejections=all(r['all_placement_edges_below_original_15m'] and r['all_original_calls_rejected'] for r in rows),
        source_sha256=sha(__file__), protected_409_source_unchanged=True, API_requests=0,
        actual_execution_frames=0, D=None, goal_complete=False))


if __name__ == '__main__':
    main()
