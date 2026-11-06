"""真实推理工作进程入口；默认旧路径，显式开启高层修复，禁止向旧目录写收据。"""
from pathlib import Path
import argparse,hashlib,json,sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--highlevel-grounding',choices=['global','candidate_local'],default='global')
    p.add_argument('--identity-comparison-facts',choices=['off','on'],default='off')
    a,remaining=p.parse_known_args();out=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    if out.exists():raise FileExistsError('preserve_existing_actual_worker_output')
    if a.highlevel_grounding=='candidate_local' or a.identity_comparison_facts=='on':
        for key in ['--skills','--incremental-skills','--rolling-intent','--explicit-contract','--fractional-interaction','--motion-evidence']:
            if key not in remaining or remaining[remaining.index(key)+1]!='on':raise ValueError('explicit_temporal_pipeline_required:'+key)
        if not prepare:
            hashes=json.loads((out.parent/'source_sha256.json').read_text())
            required=[*HERE.glob('*.py'),ROOT/'policies/candidate_grounding.py',ROOT/'work/skills_identity_evidence_20261030/common_evidence.py',ROOT/'work/skills_identity_evidence_20261030/SKILL.md']
            if any(hashes.get(f.relative_to(ROOT).as_posix())!=hashlib.sha256(f.read_bytes()).hexdigest() for f in required):
                raise ValueError('actual_highlevel_worker_not_frozen')
    sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'work/skills_temporal_candidates_20261029')]
    from assembly import install
    import run_temporal_controlled
    previous=sys.argv
    try:
        sys.argv=[sys.argv[0],*remaining]
        with install(a.highlevel_grounding,a.identity_comparison_facts=='on') as receipt:
            return run_temporal_controlled.main()
    finally:
        sys.argv=previous
        if not prepare and out.exists():
            with (out/'candidate_local_entry.json').open('x',encoding='utf-8') as f:
                json.dump(dict(version='candidate_local_worker_entry_20261030_v1',grounding_validation_mode=a.highlevel_grounding,
                    common_identity_comparison_facts=a.identity_comparison_facts=='on',original_off_path_preserved=True,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_diffusion_refinement_enabled=False),f,indent=2)


if __name__=='__main__':main()
