"""独立默认关闭入口；复用已验证共享围栏和公共提供方预算，不改当前实验。"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--quantity-evidence',choices=('off','on'),default='off')
    args,remaining=p.parse_known_args()
    if args.quantity_evidence=='on':
        for flag in ['--skills','--incremental-skills']:
            if flag not in remaining or remaining[remaining.index(flag)+1]!='on':raise ValueError('quantity_evidence_requires_incremental_skills_on')
    output=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    if output.exists():raise FileExistsError('preserve_prior_experiment')
    sys.path[:0]=[str(ROOT/'work/skills_shared_resource_20261028'),str(ROOT/'work/skills_incremental_evidence_20261025')]
    import run_capped
    import incremental_planner
    original=incremental_planner.IncrementalSkillsPlanner;argv=sys.argv
    if args.quantity_evidence=='on':
        if not prepare:
            frozen=json.loads((output.parent/'source_sha256.json').read_text())
            for f in [*HERE.glob('*.py'),HERE/'SKILL.md']:
                if frozen.get(f.relative_to(ROOT).as_posix())!=hashlib.sha256(f.read_bytes()).hexdigest():raise ValueError('quantity_evidence_runtime_not_frozen')
        from quantity_planner import QuantityEvidencePlanner
        incremental_planner.IncrementalSkillsPlanner=partial(QuantityEvidencePlanner,quantity_evidence=True)
    try:
        sys.argv=[sys.argv[0]]+remaining
        return run_capped.main()
    finally:
        sys.argv=argv;incremental_planner.IncrementalSkillsPlanner=original
        if not prepare and output.exists():
            with (output/'quantity_entry_protocol.json').open('x',encoding='utf-8') as stream:
                json.dump(dict(version='quantity_entry_20261028_v1',enabled=args.quantity_evidence=='on',
                    new_acceptance_gates=0,new_response_fields=0,extra_requests_outside_existing_budget=0,
                    common_provider_and_resource_fence_preserved=True,prose_tactics_actual_threat_certified=False),stream,indent=2)


if __name__=='__main__':main()
