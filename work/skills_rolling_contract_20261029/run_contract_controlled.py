"""独立可关闭入口；不修改任何历史冻结版本，真实运行需完整源码快照。"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--explicit-contract',choices=('off','on'),default='off')
    p.add_argument('--deduplicate-intents',choices=('off','on'),default='off')
    options,remaining=p.parse_known_args();enabled=options.explicit_contract=='on' or options.deduplicate_intents=='on'
    if enabled:
        for flag in ['--skills','--incremental-skills','--active-window-recipes','--rolling-intent']:
            if flag not in remaining or remaining[remaining.index(flag)+1]!='on':raise ValueError('contract_entry_requires_full_rolling_path')
    output=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    if output.exists():raise FileExistsError('preserve_existing_output')
    sys.path.insert(0,str(ROOT/'work/skills_rolling_intent_20261028'))
    import rolling_planner,run_rolling_controlled
    original=rolling_planner.RollingIntentPlanner;argv=sys.argv
    if enabled:
        if not prepare:
            frozen=json.loads((output.parent/'source_sha256.json').read_text())
            for path in HERE.glob('*.py'):
                if frozen.get(path.relative_to(ROOT).as_posix())!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('contract_entry_not_frozen')
        from contract_planner import ContractRollingPlanner
        rolling_planner.RollingIntentPlanner=partial(ContractRollingPlanner,
            explicit_contract=options.explicit_contract=='on',deduplicate=options.deduplicate_intents=='on')
    try:
        sys.argv=[sys.argv[0]]+remaining;return run_rolling_controlled.main()
    finally:
        sys.argv=argv;rolling_planner.RollingIntentPlanner=original
        if enabled and not prepare and output.exists():
            with (output/'explicit_contract_entry.json').open('x',encoding='utf-8') as f:
                json.dump(dict(version='rolling_contract_entry_20261029_v1',explicit_contract=options.explicit_contract,
                    deduplicate_intents=options.deduplicate_intents,original_validator_preserved=True,no_extra_requests=True),f,indent=2)


if __name__=='__main__':main()
