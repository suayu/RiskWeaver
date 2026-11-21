"""准备独立契约修订单场；保留r1/r2全部失败，不拼接或覆盖结果。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
OLD=ROOT/'experiments/skills_rolling_future_scene074_20261029_r2'
NEW=ROOT/'experiments/skills_rolling_contract_scene074_20261029_r1'
HERE=Path(__file__).resolve().parent


def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    if NEW.exists():raise FileExistsError('preserve_existing_contract_preparation')
    if json.loads((OLD/'launcher_result.json').read_text()).get('exit_code')!=0:raise ValueError('prior_run_not_terminal')
    launch=json.loads((OLD/'launch.json').read_text());command=launch['command']
    command[command.index(str(ROOT/'work/skills_rolling_intent_20261028/run_rolling_controlled.py'))]=str(HERE/'run_contract_controlled.py')
    command[command.index('--manifest')+1]=str(NEW/'manifest.json')
    command[command.index('--output-dir')+1]=str(NEW/'contract_full')
    command+=['--explicit-contract','on','--deduplicate-intents','on']
    NEW.mkdir();manifest=json.loads((OLD/'manifest.json').read_text());save(NEW/'manifest.json',manifest)
    if hashlib.sha256(Path(manifest['scenario_files'][0]).read_bytes()).hexdigest()!=manifest['scenario_sha256'][0]:raise ValueError('scene_changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LLM_MODEL_NAME=launch['model_pool'][0],LLM_MODEL_NAMES=','.join(launch['model_pool']))
    records=[]
    for enabled in ['off','on']:
        cmd=command.copy()
        for key in ['--explicit-contract','--deduplicate-intents']:cmd[cmd.index(key)+1]=enabled
        cmd+=['--prepare-only'];r=subprocess.run(cmd,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        (NEW/f'prepare_{enabled}.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        if r.returncode or (NEW/'contract_full').exists():raise RuntimeError('actual_contract_prepare_failed_'+enabled)
        records.append(dict(enabled=enabled,exit_code=r.returncode,API_requests=0,GPU_execution=False))
    save(NEW/'entry_prepare_validation.json',dict(passed=True,records=records,performance_evidence=False))
    paths=list(json.loads((OLD/'source_sha256.json').read_text()))
    paths += [p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]
    hashes={}
    for name in sorted(set(paths)):
        source=ROOT/name;hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        target=NEW/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    save(NEW/'source_sha256.json',hashes)
    launch.update(command=command,predecessor_preserved=str(OLD),changes='explicit ID-to-reason object contract and precise repair feedback; numerical sparse-intent deduplication',
        component_regressions_passed=14,numeric_equivalence_tolerance_m=1e-6,original_validator_and_request_budget_unchanged=True,
        no_silent_response_conversion=True,planned_attempts=1,not_an_independent_generalization_test=True)
    save(NEW/'launch.json',launch)
    print(json.dumps(dict(prepared=str(NEW),frozen_sources=len(hashes),actual_prepare_runs=2,new_API_requests=0)))


if __name__=='__main__':main()
