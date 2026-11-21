"""复用E1闭环的技能对照入口；不改旧运行器与旧对照配置。"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--skills',choices=('off','on'),required=True)
    parser.add_argument('--skill-level',choices=('none','protocol','analysis','full'),default='full')
    parser.add_argument('--candidate-expansion',choices=('off','on'),default='off')
    args,remaining=parser.parse_known_args()
    from experiments.comparison import run_e1_group as runner
    original=runner.group_overrides
    def overrides(*positional,**kwargs):
        # 显式百炼、统一求解器与攻击有效期；每个结果目录保留完整resolved_config。
        values=original(*positional,**kwargs)
        values += [f'sim.llm.skills.enabled={str(args.skills=="on").lower()}',
                   f'sim.llm.skills.level={args.skill_level}',
                   f'sim.llm.skills.candidate_expansion_enabled={str(args.candidate_expansion=="on").lower()}']
        return values
    runner.group_overrides=overrides
    # 独立记录真实单组协议，避免把E1五组模板摘要误当成本次实验规模。
    files=['scenario_generator.py','policies/skills_planner.py','policies/skill_registry.py',
           'policies/intent_validation.py','policies/llm_adversarial_planner.py','skills/riskweaver/manifest.json']
    frozen={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    sys.argv=[sys.argv[0]]+remaining
    try:
        runner.main()
    finally:
        if '--output-dir' in remaining:
            out=Path(remaining[remaining.index('--output-dir')+1])
            if out.exists():
                protocol=dict(provider='dashscope',skills=args.skills,level=args.skill_level,
                    candidate_expansion=args.candidate_expansion,source_sha256=frozen,
                    source_unchanged_during_run=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in frozen.items()),
                    quota_policy='retain existing user-configured free protection; no cross-provider fallback')
                (out/'skills_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
