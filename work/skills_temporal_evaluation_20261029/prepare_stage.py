"""新时序版阶段准备；真实prepare-only两臂，不创建或派发实验。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
SMOKE=ROOT/'experiments/skills_temporal_candidates_scene074_20261029_r1'
OLD=ROOT/'experiments/skills_four_arm_provider_shared_development10_20261028_r2'
sys.path[:0]=[str(ROOT/'work/skills_modelpool_budget_dev_20261022'),str(ROOT/'work/skills_heldout_validation_20261029')]
from model_pool_protocol import inspect_model_source
from validation_core import rewrite_command
from stage_core import ARMS,validate_manifest


def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--stage',choices=('development','validation'),required=True)
    p.add_argument('--current-development-result',type=Path);a=p.parse_args();folder=a.protocol
    if not folder.is_absolute() or ROOT not in folder.parents:raise ValueError('workspace_protocol_path_required')
    if folder.exists():raise FileExistsError('preserve_prior_protocol')
    if a.stage=='validation' and (a.current_development_result is None or not a.current_development_result.is_absolute()
            or ROOT not in a.current_development_result.parents or a.current_development_result==OLD):
        raise ValueError('explicit_current_temporal_development_result_required')
    previous=OLD/'source_snapshot/work/skills_provider_shared_protocol_20261028_r2'
    manifest=json.loads((previous/f'{a.stage}10.json').read_text());counts=validate_manifest(manifest,a.stage)
    for path,digest in zip(manifest['scenario_files'],manifest['scenario_sha256']):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('scenario_payload_changed')
    folder.mkdir();save(folder/f'{a.stage}10.json',manifest)
    launch=json.loads((SMOKE/'launch.json').read_text());pool=launch['model_pool']
    templates={ARMS[0]:json.loads((OLD/(ARMS[0]+'_pid.json')).read_text())['command'],ARMS[1]:launch['command']}
    prepared={};previews=[]
    for arm,template in templates.items():
        output=folder/('preview_'+arm);command=rewrite_command(template,folder/f'{a.stage}10.json',output,pool)
        prepared[arm]=dict(command=command,scope='legacy' if arm==ARMS[0] else 'whole_temporal_package')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LLM_MODEL_NAME=pool[0],LLM_MODEL_NAMES=','.join(pool))
        result=subprocess.run(command+['--prepare-only'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        (folder/(arm+'_prepare.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode or output.exists():raise RuntimeError('stage_prepare_failed:'+arm)
        previews.append(dict(arm=arm,exit_code=0,new_API_requests=0,GPU_execution=False))
    metadata=dict(version='temporal_stage_preparation_20261029_v1',stage=a.stage,
        smoke_experiment=SMOKE.relative_to(ROOT).as_posix(),
        current_temporal_development_experiment=a.current_development_result.relative_to(ROOT).as_posix() if a.current_development_result else None,
        model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),frozen_model_pool=pool,
        free_protection_operator_confirmed=True,**counts,arms=list(ARMS),automatic_launch=False,
        all_failures_and_unattempted_in_denominator=True,policy='idm',profile='off',mode='pure_dynamic',
        base_seed=42,natural_end_or_max_frames=400,physical_SDK_limit_per_scene=4,
        shared_resource=dict(gpu='0',cap_gib=14,reserve_gib=3,max_own_arms=1),
        expected_temporal_flags=dict(fractional_interaction=True,velocity_continuation=True,motion_evidence=True),
        goal_complete=False,semantic_correctness_and_threat_not_certified=True,
        remaining_scope='original_A_to_E_and_correctness_threat_static_profile_NL_target_modes_not_removed')
    save(folder/'protocol.json',metadata);save(folder/'prepared_arms.json',prepared)
    save(folder/'entry_prepare_validation.json',dict(passed=True,records=previews,not_performance_evidence=True))
    names=list(json.loads((SMOKE/'source_sha256.json').read_text()))+[p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]
    names += [(folder/n).relative_to(ROOT).as_posix() for n in ['protocol.json','prepared_arms.json',f'{a.stage}10.json']]
    hashes={}
    for name in sorted(set(names)):
        source=ROOT/name;hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        dest=folder/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    save(folder/'source_sha256.json',hashes)
    from run_stage import check_prerequisites
    ready=check_prerequisites(folder,metadata);save(folder/'readiness_at_preparation.json',ready)
    print(json.dumps(dict(stage=a.stage,prepared_only=True,frozen_sources=len(hashes),actual_prepare_runs=2,
        new_API_requests=0,dispatched_scenes=0,readiness=ready['passed'],pending_reasons=ready['reasons'])))


if __name__=='__main__':main()
