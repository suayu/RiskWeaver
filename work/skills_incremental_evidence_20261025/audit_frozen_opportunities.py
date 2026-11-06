"""零API审计原十场初始状态：自然接近误归因、现有候选缺口与新配方拒绝原因。"""
from pathlib import Path
from types import SimpleNamespace as NS
import collections
import hashlib
import json
import os
import sys
import time
import numpy as np
from omegaconf import OmegaConf

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT/'work/skills_recovery_v4_20261023'),
               str(ROOT/'work/skills_v2_compact_dev_20261020')]
from incremental_evidence import incremental_card, factual_summary
from active_window_recipes import screened_proposals
from policies.ego_profile import CandidateBuilder
from policies.intent_validation import canonical_hash, validate_world_intent
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from route_tail_preflight import route_tail_readiness


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('offline_audit_requires_hidden_GPU')
    source = ROOT/'experiments/skills_priority_v5_frozen_development10_pair_20261024_r1'
    output = ROOT/'work/skills_incremental_evidence_audit_20261025_r1'
    if output.exists():
        raise FileExistsError('preserve_existing_offline_audit')
    all_records = json.loads((source/'results.json').read_text())['records']
    inputs = json.loads((source/'inputs.json').read_text())
    config_path = ROOT/'experiments/skills_modelpool_recovery_development10_paired_20261023_r3/skills_full_shared_recovery/resolved_config.yaml'
    cfg = OmegaConf.load(config_path)
    names = list(json.loads((source/'launch.json').read_text())['source_sha256'])
    names += [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]
    names += [(source/'inputs.json').relative_to(ROOT).as_posix(), (source/'results.json').relative_to(ROOT).as_posix(), config_path.relative_to(ROOT).as_posix()]
    hashes = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(set(names))}
    output.mkdir(exist_ok=False)
    # 不调用初始化器、不加载凭据或构造SDK；只借用现有坐标/语义验证方法。
    validator = object.__new__(LLMAdversarialPlanner)
    validator._request_attack_plan = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('unexpected_API_call'))
    builder = CandidateBuilder(cfg.sim.ego_profile)
    rows = []
    for item in inputs:
        index = item['scenario_index']
        previous = next(r for r in all_records if r['arm'] == 'priority_v5' and r['scenario_index'] == index)
        context = previous['trace']['context']
        request, frame = item['request'], item['frame']
        state = request['state']
        before = canonical_hash(state)
        route = np.asarray(state['route'], float)
        env = NS(ego_state=np.asarray(frame['ego'], float), data_dict={'agent':[np.asarray(frame['agents'], float)]},
                 agent_active=np.asarray(frame['active'], bool), current_step=request['step'],
                 current_scene_id=f'scenario_{index:03d}', scenario_dict={'route':route, 'lanes':[route]},
                 dt=float(cfg.sim.dt), get_static_obstacles=lambda:[], cfg=cfg, pending_joint_trajectory=None)
        old_catalog = (context['candidates'] or context.get('route_tail_withheld_candidates') or context.get('preflight_withheld_candidates') or [])
        old_cards = [incremental_card(state, c, env.dt) for c in old_catalog]
        new, rejected = screened_proposals(env, state, builder, lambda s, c:validate_world_intent(validator, s, c))
        existing = {canonical_hash(dict(target=c['target_id'], anchors=c['anchors'])) for c in old_catalog}
        new = [c for c in new if canonical_hash(dict(target=c['target_id'], anchors=c['anchors'])) not in existing]
        new_cards = [incremental_card(state, c, env.dt) for c in new]
        combined = old_cards+new_cards
        chosen = next((c for c in old_cards if c['candidate_id'] == previous['plan'].get('candidate_id')), None)
        reasons = collections.Counter()
        for event in rejected:
            if not event['accepted']:
                reasons[event.get('screen_rejection') or event.get('semantic_validation', {}).get('reason') or event.get('reason') or 'unknown'] += 1
        # 路线尾部诊断与拒绝计数独立保存，不把新配方通过screen当作可以执行。
        tail = route_tail_readiness(env, cfg.sim.evaluation.reachability, 10) if combined else None
        rows.append(dict(scenario_index=index, step=request['step'], input_unchanged=before == canonical_hash(state),
            original_plan=previous['plan'], original_stage=previous['trace']['stage'], original_published_count=len(context['candidates']),
            old_cards=old_cards, old_selected_card=chosen,
            new_sparse_candidates=new, new_cards=new_cards, proposal_screen_audit=rejected,
            proposal_rejections=dict(reasons), route_tail_diagnostic=tail,
            original_measurement_preflight=context.get('measurement_preflight'),
            best_existing_increment_m=max((c['min_gap_difference_m'] for c in old_cards), default=None),
            best_new_recipe_increment_m=max((c['min_gap_difference_m'] for c in new_cards), default=None),
            selected_factual_summary=factual_summary(chosen) if chosen else None,
            scope='initial state only; sparse CV proxy, recipe screen is not diffusion/execution or threat proof',
            actual_attack_frames=0, D=None))
    unchanged = all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n, h in hashes.items())
    result = dict(planned_states=10, audited_states=len(rows), physical_API_requests=0, new_execution_frames=0, D=None,
        source_sha256=hashes, source_unchanged=unchanged, rows=rows,
        preflight_changed=False, existing_experiments_changed=False, new_recipes_enabled_in_runtime=False,
        route_geometry_scope='same source route-as-lanes proxy as frozen planning test',
        caution='CV reference is not a real ego-policy counterfactual; do not report measured attack improvement',
        finished_at=time.time())
    (output/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (output/'source_sha256.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = dict(audited_states=len(rows), source_unchanged=unchanged, API_requests=0,
        selected_farther_than_CV=[r['scenario_index'] for r in rows if r['old_selected_card'] and r['old_selected_card']['min_gap_difference_m'] < -1e-6],
        old_zero_catalog_new_screened=[r['scenario_index'] for r in rows if not r['old_cards'] and r['new_cards']],
        all_input_unchanged=all(r['input_unchanged'] for r in rows))
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    assert unchanged and summary['all_input_unchanged'] and len(rows) == 10


if __name__ == '__main__':main()
