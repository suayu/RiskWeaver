"""准备原留出10场两臂协议；单场未通过时仍只输出待验收状态，不派发。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
DEVELOPMENT=ROOT/'experiments/skills_four_arm_provider_shared_development10_20261028_r2'
SMOKE=ROOT/'experiments/skills_rolling_contract_scene074_20261029_r1'
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from model_pool_protocol import inspect_model_source
from validation_core import ARMS,validate_manifest,rewrite_command


def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args()
    folder=args.protocol
    if not folder.is_absolute() or ROOT not in folder.parents:raise ValueError('explicit_workspace_protocol_path_required')
    if folder.exists():raise FileExistsError('preserve_prior_protocol')
    validation=DEVELOPMENT/'source_snapshot/work/skills_provider_shared_protocol_20261028_r2/validation10.json'
    manifest=json.loads(validation.read_text());counts=validate_manifest(manifest)
    for path,digest in zip(manifest['scenario_files'],manifest['scenario_sha256']):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('original_holdout_pickle_changed')
    folder.mkdir();save(folder/'validation10.json',manifest)
    smoke_launch=json.loads((SMOKE/'launch.json').read_text());pool=smoke_launch['model_pool']
    templates={ARMS[0]:json.loads((DEVELOPMENT/(ARMS[0]+'_pid.json')).read_text())['command'],
        ARMS[1]:smoke_launch['command']}
    prepared={};previews=[]
    for arm,template in templates.items():
        output=folder/('preview_'+arm)
        command=rewrite_command(template,folder/'validation10.json',output,pool)
        prepared[arm]=dict(command=command,scope='legacy' if arm==ARMS[0] else 'complete_rolling_contract_package')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LLM_MODEL_NAME=pool[0],LLM_MODEL_NAMES=','.join(pool))
        result=subprocess.run(command+['--prepare-only'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        (folder/(arm+'_prepare.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode or output.exists():raise RuntimeError('actual_paired_prepare_failed:'+arm)
        previews.append(dict(arm=arm,exit_code=0,API_requests=0,GPU_execution=False))
    save(folder/'prepared_arms.json',prepared)
    metadata=dict(version='original_holdout10_rolling_contract_pair_20261029_v1',
        development_experiment=DEVELOPMENT.relative_to(ROOT).as_posix(),smoke_experiment=SMOKE.relative_to(ROOT).as_posix(),
        model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),frozen_model_pool=pool,
        free_protection_operator_confirmed=True,**counts,arms=list(ARMS),
        policy='idm',profile='off',mode='pure_dynamic',seed=42,natural_end_or_max_frames=400,
        physical_SDK_limit_per_scene=4,shared_resource=dict(gpu='0',cap_gib=14,reserve_gib=3,max_own_arms=1),
        automatic_launch=False,status_at_preparation='prepared_only_new_smoke_prerequisite_pending',
        original_validation_manifest_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
        runtime_claim_scope='whole_package_operational_comparison_not_skill_text_only_or_universal_noninferiority',
        remaining_scope='P1_semantics_P2_threat_and_original_A_to_E_remaining_modes_still_require_independent_evidence')
    save(folder/'protocol.json',metadata);save(folder/'entry_prepare_validation.json',dict(passed=True,previews=previews))
    paths=list(json.loads((SMOKE/'source_sha256.json').read_text()))
    paths += [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]
    paths += [(folder/name).relative_to(ROOT).as_posix() for name in ['protocol.json','validation10.json','prepared_arms.json']]
    hashes={}
    for name in sorted(set(paths)):
        source=ROOT/name;hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        dest=folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    save(folder/'source_sha256.json',hashes)
    from run_validation import check_prerequisites
    ready=check_prerequisites(folder,metadata);save(folder/'readiness_at_preparation.json',ready)
    print(json.dumps(dict(protocol=str(folder),frozen_sources=len(hashes),actual_prepare_runs=2,
        readiness=ready['passed'],pending_reasons=ready['reasons'],new_API_requests=0,dispatched_scenes=0)))


if __name__=='__main__':main()
