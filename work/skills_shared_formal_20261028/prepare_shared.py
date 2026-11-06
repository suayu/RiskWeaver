"""新目录冻结同一40预定尝试，只更换共同的资源准入和显存围栏。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'work/skills_provider_controlled_protocol_20261028_r1'
OUT=ROOT/'work/skills_provider_shared_protocol_20261028_r2'
EXP=ROOT/'experiments/skills_four_arm_provider_shared_development10_20261028_r2'
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from launch_pool_pair import verify_sources
from model_pool_protocol import inspect_model_source


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def set_flag(command,flag,value):
    if flag in command:command[command.index(flag)+1]=str(value)
    else:command.extend([flag,str(value)])


def main():
    if OUT.exists() or EXP.exists():raise FileExistsError('preserve_formal_trials')
    old=json.loads((OLD/'protocol.json').read_text());hashes=json.loads((OLD/'source_sha256.json').read_text())
    if not verify_sources(old,hashes)['verified']:raise ValueError('original_algorithm_changed')
    source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    pool=[m for m in old['frozen_model_pool'] if m not in source['disabled_models']]
    if not pool:raise ValueError('approved_pool_empty')
    OUT.mkdir()
    for name in ['development10.json','validation10.json','smoke074.json']:(OUT/name).write_bytes((OLD/name).read_bytes())
    prepared=copy.deepcopy(json.loads((OLD/'prepared_arms.json').read_text()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LLM_MODEL_NAME=pool[0],LLM_MODEL_NAMES=','.join(pool))
    for arm,row in prepared.items():
        command=row['launch_command_pending_confirmation'];command[1]=str(ROOT/'work/skills_shared_resource_20261028/run_capped.py')
        for flag,value in [('--manifest',OUT/'development10.json'),('--output-dir',EXP/arm),('--model',pool[0]),
            ('--steps',400),('--resource-cap-gib','14'),('--resource-reserve-gib','3'),('--resource-fixture','off'),('--resource-observe-sampling','off')]:
            set_flag(command,flag,value)
        start=command.index('--models')+1;end=next(i for i in range(start,len(command)) if command[i].startswith('--'))
        command[start:end]=pool
        checked=subprocess.run(command+['--prepare-only'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        row.update(prepare_result=json.loads(checked.stdout),resource_fixture=False,started=False)
    metadata=copy.deepcopy(old);metadata.update(version='provider_shared_four_arm_20261028_r2',original_protocol=str(OLD),
        frozen_model_pool=pool,model_source=source,launcher_entry=str(HERE/'launch_shared.py'))
    metadata['shared_resource']=dict(gpu='0',max_active_arms=1,cap_gib=14.,reserve_gib=3.,
        probe_experiment='experiments/skills_shared_resource_probe_20261028_r1',
        prior_user_authorization='2026-10-28 shared_probe_then_proceed',
        all_arms_same_resource_policy=True,other_user_GPU_workloads_may_change=True,
        not_uncontended_timing_benchmark=True,fixture_allowed=False,sampling_observer_sync=False)
    metadata['comparison_controls']['model_policy']='same_approved_remaining8_pool; qwen3.8-max-0902_primary; quota_and_service_fallback_disclosed'
    save(OUT/'prepared_arms.json',prepared);save(OUT/'protocol.json',metadata)
    names=set(hashes)
    for folder in [HERE,ROOT/'work/skills_shared_resource_20261028']:
        names.update(p.relative_to(ROOT).as_posix() for p in folder.glob('*') if p.is_file() and p.suffix in {'.py','.md','.json'})
    names.update((OUT/name).relative_to(ROOT).as_posix() for name in ['development10.json','validation10.json','smoke074.json','prepared_arms.json','protocol.json'])
    frozen={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(names)}
    for name in frozen:
        p=OUT/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    save(OUT/'source_sha256.json',frozen)
    print(json.dumps(dict(protocol=str(OUT),source_files=len(frozen),prepare_checks=4,planned_attempts=40,API_requests=0,experiments_started=0)))


if __name__=='__main__':main()
