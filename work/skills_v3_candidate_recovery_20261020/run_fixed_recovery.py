"""十帧固定意图集成检验入口；开关默认关闭，严格零API。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_lifecycle_dev_20261020'))
import fixed_intent_lifecycle as fixed
from joint_candidate_recovery import install_recovery


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--joint-candidate-recovery',choices=('off','on'),default='off')
    args,remaining=parser.parse_known_args()
    output=Path(remaining[remaining.index('--output-dir')+1])
    fixed.SOURCE=ROOT/'experiments/skills_v2_frozen_development10_cpu_pair_20261020_r1/skills_v2/movies/scenario_056/execution_trace.jsonl'
    restore=install_recovery(max_candidates=7,max_seconds=30.) if args.joint_candidate_recovery=='on' else lambda:None
    sys.argv=[sys.argv[0],'--duration','10']+remaining
    try:fixed.main()
    finally:
        restore()
        if output.exists():
            (output/'recovery_protocol.json').write_text(json.dumps(dict(
                joint_candidate_recovery=args.joint_candidate_recovery,additional_LLM_calls=0,
                additional_diffusion_samples=0,max_candidates=7,recovery_acceptance_budget_seconds=30.,
                scope='ten-frame fixed-intent integration and cache check; not a full episode or LLM generation coverage experiment',
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__),Path(__file__).with_name('joint_candidate_recovery.py')]}),indent=2))


if __name__=='__main__':main()
