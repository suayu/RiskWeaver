"""250冻结场景的初态静态目录审计；纯CPU、零API、零env.step。"""
import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """原子写独立审计结果，不写回原场景或任何正在运行的实验。"""
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def initial_environment(geometry, cfg):
    """复用公共FOV、快照及规划状态方法；不初始化模型、渲染器或执行器。"""
    import numpy as np
    from datasets.waymo.dataset_ctrl_sim import CtRLSimDataset
    from simulator import Simulator
    env = Simulator.__new__(Simulator)
    env.cfg = cfg
    env.t = 0
    env.dt = float(cfg.sim.dt)
    env.current_scene_id = geometry['_audit_filename']
    env.nearby_distance = 35
    env.scenario_dict = copy.deepcopy({k:v for k,v in geometry.items() if k != '_audit_filename'})
    agents = env.scenario_dict['agents']
    env.ego_trajectory = agents[-1].copy()
    env.ego_state = env.ego_trajectory[0].copy()
    types = env.scenario_dict['agent_types']
    keep = types[:-1, 1] == 1 if cfg.sim.simulate_vehicles_only else np.ones(len(types)-1, dtype=bool)
    env.scenario_dict['agents'] = agents[:-1][keep]
    env.scenario_dict['agent_types'] = types[:-1][keep]
    host = NS(cfg=cfg.ctrl_sim.dataset, POS_X_IDX=CtRLSimDataset.POS_X_IDX,
        POS_Y_IDX=CtRLSimDataset.POS_Y_IDX, HEAD_IDX=CtRLSimDataset.HEAD_IDX)
    local = dict(center=env.ego_state[:2].copy(), yaw=env.ego_state[4].copy())
    env.agent_active = CtRLSimDataset.get_agent_mask(host,
        copy.deepcopy(env.scenario_dict['agents'][:, :, :CtRLSimDataset.HEAD_IDX+1]), local)[:, 0]
    env.left_scene = np.zeros_like(env.agent_active, dtype=bool)
    env.data_dict = dict(agent=[env.scenario_dict['agents'][:, 0]],
        agent_active_history=[env.agent_active.copy()])
    env.get_static_obstacles = lambda: []
    return env


def verify_twenty(cfg, manifest):
    """先与历史真实初态逐数组、逐JSON核对，失败即不进入250场扫描。"""
    import numpy as np
    groups = [
        ('work/skills_candidate_local_development_protocol_20261030_r4',
         'experiments/skills_temporal_development10_paired_20261029_r1/manifest.json'),
        ('work/skills_candidate_local_holdout_protocol_20261030_r1',
         'work/skills_candidate_local_holdout_protocol_20261030_r1/validation10.json'),
    ]
    checks = []
    for protocol, path in groups:
        samples = json.loads((ROOT/protocol/'inputs.json').read_text())
        previous = json.loads((ROOT/path).read_text())
        assert len(samples) == 10
        for sample, filename, digest in zip(samples, previous['scenario_files'], previous['scenario_sha256']):
            assert sha(filename) == digest
            index = sample['scene']
            assert manifest['scenario_files'][index] == filename
            assert manifest['scenario_sha256'][index] == digest
            geometry = pickle.loads(Path(filename).read_bytes())
            geometry['_audit_filename'] = Path(filename).name
            env = initial_environment(geometry, cfg)
            state = env.get_state_for_planning()
            frame = sample['frame']
            item = dict(scene=index, source_step=sample['source_step'],
                ego_equal=bool(np.array_equal(env.ego_state, np.asarray(frame['ego']))),
                agents_equal=bool(np.array_equal(env.data_dict['agent'][-1], np.asarray(frame['agents']))),
                active_equal=bool(np.array_equal(env.agent_active, np.asarray(frame['active']))),
                planning_state_exact=state == sample['state'])
            checks.append(item)
    return dict(expected=20, checked=len(checks),
        passed=len(checks) == 20 and all(all(c[k] for k in ['ego_equal','agents_equal','active_equal','planning_state_exact']) and c['source_step']==0 for c in checks),
        checks=checks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU_only_required')
    from omegaconf import OmegaConf
    from policies.ego_profile import CandidateBuilder
    from policies.static_attack_catalog import build_static_combinations
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    manifest_path = ROOT/'experiments/scenario_dreamer_ldm_expand250_20260928_r1/riskweaver_frozen_250_scenes.json'
    config_path = ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml'
    manifest = json.loads(manifest_path.read_text())
    cfg = OmegaConf.load(config_path)
    assert manifest['scenario_count'] == len(manifest['scenario_files']) == len(manifest['scenario_sha256']) == 250
    assert len(set(manifest['scenario_sha256'])) == 250
    assert cfg.sim.simulate_vehicles_only and cfg.sim.dt == .1
    a.output.mkdir(exist_ok=False)
    baseline = ROOT/'work/skills_hybrid_closed_loop2_protocol_20261102_r1/source_sha256.json'
    protected = json.loads(baseline.read_text())
    def integrity():
        return [name for name,digest in protected.items() if sha(ROOT/name) != digest]
    if integrity():
        raise ValueError('running_409_source_changed_before_CPU_audit')
    protocol = dict(scope='all_250_initial_static_catalogs_forced_probe_not_hybrid_dynamic_selection',
        planned=250, API_requests=0, env_step_calls=0, D=None,
        manifest_sha256=sha(manifest_path), config_sha256=sha(config_path),
        protected_source_count=len(protected), source_sha256=protected,
        audit_source_sha256=sha(__file__),
        dynamic_catalogs_rebuilt=False, actual_attack_acceptance_or_coverage_certified=False)
    save(a.output/'protocol.json', protocol)
    # 保存审计自身与冻结清单；不把未来修改的同名脚本误认为本轮实现。
    (a.output/'audit_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'frozen_manifest.json').write_bytes(manifest_path.read_bytes())
    started = time.time()
    try:
        equivalence = verify_twenty(cfg, manifest)
        save(a.output/'initial_state_equivalence20.json', equivalence)
        if not equivalence['passed']:
            raise ValueError('initial_state_projection_disagrees_with_real_frozen_inputs')
        if a.prepare_only:
            save(a.output/'stage_result.json', dict(status='prepared_only', scanned=0, equivalence_passed=True, API_requests=0))
            return
        planner = LLMAdversarialPlanner(model_name='qwen3.8-2.4t-a95b',
            model_names=['qwen3.8-2.4t-a95b'], provider='dashscope', client=NS())
        planner.use_obstacles = True
        def forbidden(*args, **kwargs):
            raise AssertionError('CPU_catalog_audit_must_not_call_model')
        planner._request_attack_plan = forbidden
        builder = CandidateBuilder(cfg.sim.ego_profile)
        rows = []
        for index, (filename, digest) in enumerate(zip(manifest['scenario_files'], manifest['scenario_sha256'])):
            if sha(__file__) != protocol['audit_source_sha256'] or sha(manifest_path) != protocol['manifest_sha256']:
                raise ValueError('audit_code_or_manifest_changed')
            if integrity():
                raise ValueError('running_409_source_changed_during_CPU_audit')
            if sha(filename) != digest:
                raise ValueError('frozen_scene_changed')
            geometry = pickle.loads(Path(filename).read_bytes())
            geometry['_audit_filename'] = Path(filename).name
            env = initial_environment(geometry, cfg)
            catalog, audit = build_static_combinations(env, planner, builder)
            row = dict(scene=index, scenario_path=filename, scenario_sha256=digest,
                active_background_count=int(env.agent_active.sum()),
                catalog_count=len(catalog), catalog_available=bool(catalog),
                candidate_audit=audit, static_catalog=catalog,
                rejection_counts=dict(Counter(x.get('reason') for x in audit if not x.get('accepted'))),
                accepted_LLM_plan=None, actual_execution_frames=0, D=None)
            rows.append(row)
            save(a.output/f'scene{index:03d}.json', row)
            save(a.output/'progress.json', dict(planned=250, scanned=len(rows),
                with_static_catalog=sum(r['catalog_available'] for r in rows),
                missing_scenes=[r['scene'] for r in rows if not r['catalog_available']],
                elapsed_seconds=time.time()-started))
        bad = integrity()
        if bad:
            raise ValueError('running_409_source_changed_at_CPU_audit_end')
        save(a.output/'stage_result.json', dict(status='completed', planned=250, scanned=len(rows),
            with_static_catalog=sum(r['catalog_available'] for r in rows),
            missing_scenes=[r['scene'] for r in rows if not r['catalog_available']],
            rejection_counts=dict(sum((Counter(r['rejection_counts']) for r in rows), Counter())),
            elapsed_seconds=time.time()-started, API_requests=0, new_execution_frames=0, D=None,
            protected_409_source_unchanged=True, goal_complete=False))
    except Exception as error:
        save(a.output/'stage_result.json', dict(status='failed', error_type=type(error).__name__,
            error=str(error), elapsed_seconds=time.time()-started, API_requests=0, D=None, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
