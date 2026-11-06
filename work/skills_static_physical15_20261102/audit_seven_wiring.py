"""七个原目录缺口的完整静态请求/复查接线；回复为替身，0 API/0 env.step。"""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'work/skills_catalog250_20261102'),
               str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from audit import initial_environment
from launch_pool_pair import verify_sources
from policies.ego_profile import clean_json
from policies.intent_validation import canonical_hash
from work.skills_static_physical15_20261102.planner import make_static, RULE_PATH


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    content = json.dumps(clean_json(value), ensure_ascii=False, allow_nan=False, indent=2)
    pending = path.with_suffix('.pending')
    pending.write_text(content, encoding='utf-8')
    pending.replace(path)


def fixture_planner():
    spec = importlib.util.spec_from_file_location('seven_wiring_test_fixture',
        ROOT/'work/hybrid_public_integration_20261101/test_public_hybrid.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, planner, _, requests = module.setup(False)
    request = planner._request_attack_plan
    def selected(*args, **kwargs):
        reasoning, response = request(*args, **kwargs)
        response['scene_analysis'] = 'DETERMINISTIC_WIRING_FIXTURE_NOT_LLM; choose the first published static combination.'
        return reasoning, response
    planner._request_attack_plan = selected
    planner.last_successful_model_name = None
    planner.last_attempted_model_names = []
    planner.last_model_failures = []
    def forbidden(*args, **kwargs):
        raise AssertionError('real_API_is_forbidden_in_wiring_audit')
    planner.client = NS(chat=NS(completions=NS(create=forbidden)))
    return planner, requests


def main(output=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU_only_required')
    from omegaconf import OmegaConf, open_dict
    out = Path(output) if output is not None else HERE/'seven_wiring_r1'
    if out.exists():
        raise FileExistsError('preserve_existing_wiring_results')
    prior = ROOT/'work/skills_route_static_fallback_20261102/all250_physical15_r1'
    stage = json.loads((prior/'stage_result.json').read_text())
    if stage['status'] != 'completed' or stage['scanned'] != 250:
        raise ValueError('full_catalog_audit_missing')
    scenes = stage['recovered_union_scenes']
    if scenes != [91, 100, 129, 211, 238, 241, 246]:
        raise ValueError('do_not_reselect_seven_cases')
    manifest_path = ROOT/'work/skills_catalog250_20261102/all250_r1/frozen_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    cfg_path = ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml'
    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg.sim.llm.skills):
        cfg.sim.llm.skills.source_fact_contract_enabled = True
    protocol_path = ROOT/'work/skills_hybrid_closed_loop2_deepflash_protocol_20261102_r4'
    protected = json.loads((protocol_path/'source_sha256.json').read_text())
    source_protocol = json.loads((protocol_path/'protocol.json').read_text())
    if not verify_sources(source_protocol, protected)['verified']:
        raise ValueError('protected_source_drift')
    own = {p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py', '.md') or p.name == 'manifest.json'}
    out.mkdir()
    for name in own:
        (out/('source_'+name)).write_bytes((HERE/name).read_bytes())
    (out/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding='utf-8')
    (out/'frozen250_manifest.json').write_bytes(manifest_path.read_bytes())
    write(out/'protected_source_sha256.json', protected)
    write(out/'protocol.json', dict(planned_scenes=scenes, source_sha256=own,
        manifest_sha256=sha(manifest_path), config_sha256=sha(cfg_path),
        configuration_change='source_fact_contract_enabled=True; geometry unchanged',
        resolved_config_sha256=sha(out/'resolved_config.yaml'),
        previous_catalog_stage_sha256=sha(prior/'stage_result.json'),
        physical_API_requests=0, env_step_calls=0, D=None, collision=None,
        scope='public_static_branch_wiring_only; not_fresh_LLM_or_full_temporal_factory', goal_complete=False))
    rows = []
    attempted = 0
    try:
        for index in scenes:
            if any(sha(HERE/name) != digest for name, digest in own.items()):
                raise ValueError('wiring_source_changed')
            filename = manifest['scenario_files'][index]
            if sha(filename) != manifest['scenario_sha256'][index]:
                raise ValueError('scenario_content_changed')
            attempted += 1
            geometry = pickle.loads(Path(filename).read_bytes())
            geometry['_audit_filename'] = Path(filename).name
            env = initial_environment(geometry, cfg)
            def forbidden(*args, **kwargs):
                raise AssertionError('env_step_is_forbidden_in_wiring_audit')
            env.step = forbidden
            state = env.get_state_for_planning()
            variants = {}
            for enabled, name in [(False, 'original'), (True, 'physical15')]:
                planner, calls = fixture_planner()
                engine = make_static(cfg.sim, call_budget=4, enabled=enabled)
                # 独立目录预览与plan内部构造必须得到同一候选，不能只修其中一条路径。
                context = engine.build_context(env, state, planner)
                plan = engine.plan(planner, env, state, ['动态目录不可用时选择已发布静态组合'])
                variants[name] = dict(catalog_count=len(context['obstacle_candidates']),
                    plan=plan, requests=calls, trace=planner.last_trace,
                    mock_request_count=len(calls), physical_API_requests=0,
                    accepted_fixture=bool(plan and plan.get('obstacle_plan')),
                    not_LLM_performance_evidence=True)
                if enabled:
                    previous = json.loads((prior/f'scene{index:03d}.json').read_text())
                    assert canonical_hash(context['obstacle_candidates']) == canonical_hash(previous['static_catalog'])
                    assert len(calls) == 1
                    assert calls[0]['context']['static_clearance_policy']['fallback_initial_ego_OBB_margin_m'] == 15.
                    assert any(d['path'] == RULE_PATH for d in calls[0]['knowledge']['documents'])
                    assert canonical_hash(calls[0]['context']['obstacle_candidates']) == canonical_hash(context['obstacle_candidates'])
                    assert planner.last_trace['static_physical15']['final_checks'][-1]['accepted']
                    assert variants[name]['accepted_fixture']
                else:
                    assert not context['obstacle_candidates'] and not calls
                    assert not variants[name]['accepted_fixture']
                assert canonical_hash(env.get_state_for_planning()) == canonical_hash(state)
            row = dict(scene=index, scenario_sha256=manifest['scenario_sha256'][index],
                       source_state_sha256=canonical_hash(state), variants=variants,
                       D=None, collision=None, execution_frames=0, physical_API_requests=0)
            write(out/f'scene{index:03d}.json', row)
            rows.append(row)
        integrity = verify_sources(source_protocol, protected)
        write(out/'runtime_source_integrity.json', integrity)
        if not integrity['verified']:
            raise ValueError('protected_source_changed_during_audit')
        write(out/'stage_result.json', dict(status='completed', planned=7, attempted=attempted,
            completed=len(rows),
            original_catalog_available=sum(r['variants']['original']['catalog_count'] > 0 for r in rows),
            physical15_catalog_available=sum(r['variants']['physical15']['catalog_count'] > 0 for r in rows),
            fixture_selected_and_revalidated=sum(r['variants']['physical15']['accepted_fixture'] for r in rows),
            fresh_LLM_successes=None, physical_API_requests=0, env_step_calls=0,
            D=None, collision=None, goal_complete=False))
    except Exception as error:
        write(out/'stage_result.json', dict(status='failed', planned=7, attempted=attempted,
            completed=len(rows), unattempted=7-attempted, error_type=type(error).__name__, error=str(error),
            physical_API_requests=0, env_step_calls=0, D=None, collision=None, goal_complete=False))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.output)
