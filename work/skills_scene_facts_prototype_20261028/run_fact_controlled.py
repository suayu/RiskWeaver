"""事实层的独立真实入口，默认关闭；不修改原252源码协议及正在排队的实验。"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scene-fact-grounding',choices=('off','on'),default='off')
    options,remaining=parser.parse_known_args()
    if options.scene_fact_grounding=='on':
        for flag in ['--incremental-skills','--skills']:
            if flag not in remaining or remaining[remaining.index(flag)+1]!='on':
                raise ValueError('scene_fact_layer_requires_skills_and_incremental_on')
    output=Path(remaining[remaining.index('--output-dir')+1])
    prepare='--prepare-only' in remaining
    if output.exists():
        raise FileExistsError('preserve_prior_experiment')
    sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_provider_compat_20261028'),str(ROOT/'work/skills_incremental_evidence_20261025')]
    import run_controlled
    import incremental_planner
    original=incremental_planner.IncrementalSkillsPlanner
    if options.scene_fact_grounding=='on':
        from fact_planner import SceneFactPlanner
        if not prepare:
            frozen=json.loads((output.parent/'source_sha256.json').read_text())
            for p in HERE.glob('*.py'):
                if frozen.get(p.relative_to(ROOT).as_posix())!=hashlib.sha256(p.read_bytes()).hexdigest():
                    raise ValueError('scene_fact_runtime_sources_not_frozen')
        incremental_planner.IncrementalSkillsPlanner=partial(SceneFactPlanner,fact_grounding=True)
    argv=sys.argv
    try:
        sys.argv=[sys.argv[0]]+remaining
        return run_controlled.main()
    finally:
        sys.argv=argv
        incremental_planner.IncrementalSkillsPlanner=original
        if not prepare and output.exists():
            with (output/'scene_fact_entry_protocol.json').open('x',encoding='utf-8') as stream:
                json.dump(dict(version='scene_fact_entry_20261028_v1',enabled=options.scene_fact_grounding=='on',
                    common_provider_and_physical_budget_preserved=True,extra_unbudgeted_requests=0,
                    prose_and_tactics_not_certified=True),stream,indent=2)


if __name__=='__main__':
    main()
