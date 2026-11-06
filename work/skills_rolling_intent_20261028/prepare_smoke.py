"""冻结单场诊断并真实prepare关/开入口；只准备，不发送请求或启动GPU。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
OLD=ROOT/'experiments/skills_four_arm_provider_shared_development10_20261028_r2'
NEW=ROOT/'experiments/skills_rolling_future_scene074_20261029_r2'


def save(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)


def main():
    if not json.loads((OLD/'launcher_result.json').read_text()).get('acceptance_passed'):raise ValueError('previous_four_arm_not_accepted')
    if NEW.exists():raise FileExistsError('preserve_old_smoke_preparation')
    manifest=json.loads((OLD/'manifest.json').read_text());index=manifest['original_scenario_indices'].index(74)
    for key in ['scenario_files','scenario_sha256','original_scenario_indices']:manifest[key]=[manifest[key][index]]
    manifest.update(scenario_count=1,comparison_scene_count=1,selection_basis='fixed_074_human_reported_temporal_diagnostic_not_performance_selection')
    if hashlib.sha256(Path(manifest['scenario_files'][0]).read_bytes()).hexdigest()!=manifest['scenario_sha256'][0]:raise ValueError('frozen_scene_changed')
    NEW.mkdir();save(NEW/'manifest.json',manifest)
    command=json.loads((OLD/'skills_full_shared_recovery_pid.json').read_text())['command']
    command[command.index(str(ROOT/'work/skills_shared_resource_20261028/run_capped.py'))]=str(ROOT/'work/skills_rolling_intent_20261028/run_rolling_controlled.py')
    begin=command.index('--models')+1;end=command.index('--policy')
    approved=['deepseek-v4.1-flash-1017','deepseek-v4.1-flash','glm-5.3','deepseek-v4-pro-0813','qwen3.8-2.4t-a95b','qwen3.8-max','qwen3.7-flash']
    command[begin:end]=approved;command[command.index('--model')+1]=approved[0]
    command[command.index('--manifest')+1]=str(NEW/'manifest.json')
    command[command.index('--output-dir')+1]=str(NEW/'rolling_full')
    command+=['--rolling-intent','on']
    preparations=[]
    for enabled in ['off','on']:
        cmd=command.copy();cmd[cmd.index('--rolling-intent')+1]=enabled;cmd+=['--prepare-only']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LLM_MODEL_NAME=approved[0],LLM_MODEL_NAMES=','.join(approved))
        r=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        (NEW/f'prepare_{enabled}.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        if r.returncode:raise RuntimeError(f'actual_prepare_{enabled}_failed')
        if (NEW/'rolling_full').exists():raise ValueError('prepare_created_runtime_output')
        preparations.append(dict(enabled=enabled,exit_code=r.returncode,requests=0,GPU_execution=False))
    save(NEW/'entry_prepare_validation.json',dict(preparations=preparations,passed=True,not_closed_loop_evidence=True))
    paths=list(json.loads((OLD/'source_sha256.json').read_text()))
    for folder in ['work/skills_rolling_intent_20261028','work/skills_quantity_evidence_20261028','work/skills_rolling_smoke_launch_20261029']:
        paths.extend(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.md'} and '__pycache__' not in p.parts)
    hashes={}
    for name in sorted(set(paths)):
        p=ROOT/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        dest=NEW/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    save(NEW/'source_sha256.json',hashes)
    save(NEW/'launch.json',dict(command=command,scenario=74,planned_attempts=1,seed=42,steps=400,physical_SDK_limit=4,
        free_protection_confirmed=True,provider='dashscope',model_pool=approved,shared_GPU='0',cap_gib=14,reserve_gib=3,
        original_solver_weights_joint_samples_safety_D_unchanged=True,changes='ego hypotheses, remaining accepted intent, sparse candidate construction and skill',
        not_same_model_causal_comparison_with_previous_four_arm=True,goal_complete=False))
    print(json.dumps(dict(prepared=str(NEW),sources=len(hashes),actual_prepare_runs=2,new_API_requests=0)))


if __name__=='__main__':main()
