"""可选Realtime闭环入口，复用原真实运行器及其预算、资源和安全围栏。"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
HERE=ROOT/'work/realtime_planner_20261104'


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--realtime-transport',choices=['off','on'],default='off')
    parser.add_argument('--base-worker',required=True,type=Path)
    options,remaining=parser.parse_known_args()
    allowed=[ROOT/'work/skills_shared_resource_20261028/run_capped.py',
             ROOT/'work/skills_candidate_local_runtime_20261030/run_worker.py']
    if options.base_worker.resolve() not in allowed:
        raise ValueError('unapproved_base_worker')
    output=Path(remaining[remaining.index('--output-dir')+1])
    prepare='--prepare-only' in remaining
    if output.exists():
        raise FileExistsError('preserve_prior_output')
    if options.realtime_transport=='on' and not prepare:
        hashes=json.loads((output.parent/'source_sha256.json').read_text())
        for name in ['transport.py','runtime.py','run_worker.py']:
            path=HERE/name
            if hashes.get(path.relative_to(ROOT).as_posix())!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('unfrozen_realtime_runtime')
    sys.path[:0]=[str(ROOT),str(options.base_worker.parent)]
    from work.realtime_planner_20261104.runtime import install
    previous=sys.argv
    try:
        sys.argv=[str(options.base_worker),*remaining]
        with install(output,options.realtime_transport=='on' and not prepare):
            runpy.run_path(str(options.base_worker),run_name='__main__')
    finally:
        sys.argv=previous
        if not prepare and output.exists():
            with (output/'realtime_entry.json').open('x',encoding='utf-8') as file:
                json.dump({'enabled':options.realtime_transport=='on','base_worker':str(options.base_worker),
                           'timeout_seconds':180,'max_tokens':8192,'automatic_retries':0,
                           'outer_physical_budget_unchanged':True,'JSON_mode':'prompt_and_local_validation'},file,indent=2)


if __name__=='__main__':
    main()
