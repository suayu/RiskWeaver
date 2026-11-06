"""复查无静态目录的全部场景，计算初态动静态候选并集；无LLM或执行。"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

from audit import ROOT, initial_environment, save, sha
from dynamic_probe import build_context


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU_only_required')
    from omegaconf import OmegaConf
    home = Path(__file__).resolve().parent
    static = home/'all250_r1'
    terminal = json.loads((static/'stage_result.json').read_text())
    assert terminal['status'] == 'completed' and terminal['scanned'] == 250
    evidence = json.loads((home/'dynamic_equivalence10_r1/stage_result.json').read_text())
    assert evidence['checked'] == 10 and evidence['exact_catalog_equal']
    protocol = json.loads((static/'protocol.json').read_text())
    protected = protocol['source_sha256']
    sources = {name:sha(home/name) for name in ['audit.py','dynamic_probe.py','union.py']}
    def verify():
        return (all(sha(ROOT/name) == digest for name,digest in protected.items())
            and all(sha(home/name) == digest for name,digest in sources.items()))
    assert verify()
    cfg = OmegaConf.load(ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
    manifest = json.loads((static/'frozen_manifest.json').read_text())
    a.output.mkdir(exist_ok=False)
    save(a.output/'protocol.json',dict(fixed_denominator=250, scope='initial_candidate_union_not_LLM_or_execution',
        static_audit_terminal_sha256=sha(static/'stage_result.json'),
        dynamic_equivalence_sha256=sha(home/'dynamic_equivalence10_r1/stage_result.json'),
        protected_source_sha256=protected, audit_sources=sources,
        dynamic_rebuild_only_where_static_absent=True, API_requests=0, env_step_calls=0))
    for name in sources:
        (a.output/('source_'+name)).write_bytes((home/name).read_bytes())
    rows = []
    started = time.time()
    try:
        for index in range(250):
            old = json.loads((static/f'scene{index:03d}.json').read_text())
            row = dict(scene=index, static_catalog_available=old['catalog_available'],
                static_catalog_count=old['catalog_count'], dynamic_catalog_rebuilt=False,
                dynamic_catalog_count=None, dynamic_catalog_available=None,
                candidate_union_available=old['catalog_available'], highlevel_acceptance=None,
                actual_execution_frames=0, D=None)
            if not old['catalog_available']:
                assert verify()
                filename = manifest['scenario_files'][index]
                assert sha(filename) == manifest['scenario_sha256'][index]
                geometry = pickle.loads(Path(filename).read_bytes())
                geometry['_audit_filename'] = Path(filename).name
                env = initial_environment(geometry, cfg)
                context = build_context(cfg, env, env.get_state_for_planning())
                row.update(dynamic_catalog_rebuilt=True, dynamic_catalog_count=len(context['candidates']),
                    dynamic_catalog_available=bool(context['candidates']),
                    candidate_union_available=bool(context['candidates']))
                save(a.output/f'scene{index:03d}_dynamic_context.json', context)
            rows.append(row)
            save(a.output/'results.partial.json',dict(planned=250, rows=rows))
        assert verify()
        save(a.output/'stage_result.json',dict(status='completed', planned=250, scanned=len(rows),
            static_available=sum(r['static_catalog_available'] for r in rows),
            dynamic_rebuilt=sum(r['dynamic_catalog_rebuilt'] for r in rows),
            recovered_by_dynamic=sum(r['dynamic_catalog_available'] is True for r in rows),
            candidate_union_available=sum(r['candidate_union_available'] for r in rows),
            both_empty_scenes=[r['scene'] for r in rows if not r['candidate_union_available']],
            rows=rows, protected_409_source_unchanged=True,
            elapsed_seconds=time.time()-started, API_requests=0, new_execution_frames=0, D=None,
            accepted_LLM_plan_coverage=None, goal_complete=False))
    except Exception as error:
        save(a.output/'stage_result.json',dict(status='failed', attempted=len(rows), planned=250,
            unattempted=250-len(rows), error_type=type(error).__name__, error=str(error), rows=rows,
            API_requests=0, actual_execution_frames=0, D=None, goal_complete=False))
        raise


if __name__ == '__main__':
    main()
