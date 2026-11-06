"""在独立进程复算初态动态目录；只用于静态缺口的候选并集审计。"""
import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace as NS

from audit import ROOT, initial_environment, save, sha

sys.path[:0] = [str(ROOT), str(ROOT/'work/skills_temporal_candidates_20261029'),
    str(ROOT/'work/skills_identity_evidence_20261030')]


def build_context(cfg, env, state):
    """沿用真实时序工厂的候选开关及原分数时刻验证器，不发请求。"""
    from temporal_planner import TemporalCandidatePlanner
    from temporal_intents import guarded_fractional_validator
    from common_evidence import install_common_evidence
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    planner = LLMAdversarialPlanner(model_name='qwen3.8-2.4t-a95b',
        model_names=['qwen3.8-2.4t-a95b'], provider='dashscope', client=NS())
    planner.use_obstacles = False
    planner.allow_trajectory = True
    def forbidden(*args, **kwargs):
        raise AssertionError('dynamic_catalog_probe_must_not_call_model')
    planner._request_attack_plan = forbidden
    engine = TemporalCandidatePlanner(cfg.sim, 4, extra_recipes=True, rolling_intent=True,
        explicit_contract=True, deduplicate=True, fractional_interaction=True,
        velocity_continuation=True, motion_evidence=True)
    install_common_evidence(engine)
    original = planner._validate_attack_plan
    planner._validate_attack_plan = guarded_fractional_validator(original, [])
    try:
        return engine.build_context(env, copy.deepcopy(state), planner)
    finally:
        planner._validate_attack_plan = original


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU_only_required')
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
    samples = json.loads((ROOT/'work/skills_hybrid_rag_highlevel_review10_protocol_20261101_r1/inputs.json').read_text())
    a.output.mkdir(exist_ok=False)
    rows = []
    # 所有十个复核初态都核对候选ID、顺序和全部候选字段，不能仅检查“非空”。
    for sample in samples:
        path = sample['scenario_path']
        assert sha(path) == sample['scenario_sha256']
        geometry = pickle.loads(Path(path).read_bytes())
        geometry['_audit_filename'] = Path(path).name
        env = initial_environment(geometry, cfg)
        context = build_context(cfg, env, sample['state'])
        expected = sample['contexts']['temporal_original']['candidates']
        actual = context['candidates']
        row = dict(scene=sample['scene'], expected_candidates=len(expected),
            actual_candidates=len(actual), exact_catalog_equal=actual == expected,
            expected_ids=[c['candidate_id'] for c in expected],
            actual_ids=[c['candidate_id'] for c in actual])
        rows.append(row)
        save(a.output/f'scene{sample["scene"]:03d}.json', dict(check=row, actual=context, expected_candidates=expected))
    save(a.output/'stage_result.json', dict(status='completed', checked=len(rows),
        exact_catalog_equal=all(r['exact_catalog_equal'] for r in rows), checks=rows,
        source_sha256=sha(__file__), API_requests=0, actual_execution_frames=0, D=None))


if __name__ == '__main__':
    main()
