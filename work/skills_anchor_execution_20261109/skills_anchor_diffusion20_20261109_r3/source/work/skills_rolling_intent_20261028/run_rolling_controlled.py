"""独立实验入口；rolling关闭时复用原路径，开启需独立冻结，不能混入已运行四组。"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--rolling-intent',choices=('off','on'),default='off')
    args,remaining=p.parse_known_args()
    if args.rolling_intent=='on':
        for flag in ['--skills','--incremental-skills','--active-window-recipes']:
            if flag not in remaining or remaining[remaining.index(flag)+1]!='on':raise ValueError('rolling_requires_full_incremental_chain')
    output=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    if output.exists():raise FileExistsError('preserve_prior_experiment')
    sys.path[:0]=[str(ROOT/'work/skills_shared_resource_20261028'),str(ROOT/'work/skills_incremental_evidence_20261025')]
    import run_capped,incremental_planner
    original=incremental_planner.IncrementalSkillsPlanner;argv=sys.argv
    if args.rolling_intent=='on':
        if not prepare:
            frozen=json.loads((output.parent/'source_sha256.json').read_text())
            required=[*HERE.glob('*.py'),HERE/'SKILL.md',* (ROOT/'work/skills_quantity_evidence_20261028').glob('*.py')]
            for f in required:
                if frozen.get(f.relative_to(ROOT).as_posix())!=hashlib.sha256(f.read_bytes()).hexdigest():raise ValueError('rolling_runtime_not_frozen')
        from rolling_planner import RollingIntentPlanner
        incremental_planner.IncrementalSkillsPlanner=partial(RollingIntentPlanner,rolling_intent=True)
    try:
        sys.argv=[sys.argv[0]]+remaining;return run_capped.main()
    finally:
        sys.argv=argv;incremental_planner.IncrementalSkillsPlanner=original
        if not prepare and output.exists():
            with (output/'rolling_entry_protocol.json').open('x',encoding='utf-8') as stream:
                json.dump(dict(version='rolling_entry_20261029_v1',enabled=args.rolling_intent=='on',
                    new_response_fields=0,safety_and_diffusion_changed=False,original_budget_preserved=True),stream,indent=2)


if __name__=='__main__':main()
