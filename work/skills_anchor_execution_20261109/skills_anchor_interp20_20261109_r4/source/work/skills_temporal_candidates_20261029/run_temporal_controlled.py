"""三项高层时序改动独立开关；关闭时沿用契约修订入口，不修改原始模块。"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(add_help=False)
    for flag in ['fractional-interaction','velocity-continuation','motion-evidence']:p.add_argument('--'+flag,choices=('off','on'),default='off')
    options,remaining=p.parse_known_args();flags={k:v=='on' for k,v in vars(options).items()};enabled=any(flags.values())
    if enabled:
        for flag in ['--skills','--incremental-skills','--rolling-intent','--explicit-contract']:
            if flag not in remaining or remaining[remaining.index(flag)+1]!='on':raise ValueError('temporal_entry_requires_explicit_contract_rolling_path')
    output=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    if output.exists():raise FileExistsError('preserve_prior_temporal_experiment')
    sys.path.insert(0,str(ROOT/'work/skills_rolling_contract_20261029'))
    import contract_planner,run_contract_controlled
    old=contract_planner.ContractRollingPlanner;argv=sys.argv
    if enabled:
        if not prepare:
            frozen=json.loads((output.parent/'source_sha256.json').read_text())
            for path in HERE.glob('*.py'):
                if frozen.get(path.relative_to(ROOT).as_posix())!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('temporal_runtime_not_frozen')
        from temporal_planner import TemporalCandidatePlanner
        contract_planner.ContractRollingPlanner=partial(TemporalCandidatePlanner,**flags)
    try:
        sys.argv=[sys.argv[0]]+remaining;return run_contract_controlled.main()
    finally:
        sys.argv=argv;contract_planner.ContractRollingPlanner=old
        if enabled and not prepare and output.exists():
            with (output/'temporal_entry_protocol.json').open('x',encoding='utf-8') as f:json.dump(dict(version='temporal_entry_20261029_v1',
                flags=flags,interaction_radius_m=15.,other_validation_and_diffusion_unchanged=True),f,indent=2)


if __name__=='__main__':main()
