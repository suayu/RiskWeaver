"""准备时序候选单场诊断，保留契约修订正/负证据，不启动留出集。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
OLD=ROOT/'experiments/skills_rolling_contract_scene074_20261029_r1'
NEW=ROOT/'experiments/skills_temporal_candidates_scene074_20261029_r1'


def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    sys.path.insert(0,str(ROOT/'work/skills_protocol_preparation_20261025'))
    from audit_four_arm import smoke_prerequisite
    prior=smoke_prerequisite(OLD,'contract_full',74)
    if not prior['passed']:raise ValueError('predecessor_runtime_smoke_not_passed')
    if NEW.exists():raise FileExistsError('preserve_previous_temporal_preparation')
    metadata=json.loads((OLD/'launch.json').read_text());command=metadata['command']
    command[command.index(str(ROOT/'work/skills_rolling_contract_20261029/run_contract_controlled.py'))]=str(HERE/'run_temporal_controlled.py')
    command[command.index('--manifest')+1]=str(NEW/'manifest.json');command[command.index('--output-dir')+1]=str(NEW/'temporal_full')
    switches=['--fractional-interaction','--velocity-continuation','--motion-evidence']
    for flag in switches:command += [flag,'on']
    manifest=json.loads((OLD/'manifest.json').read_text());NEW.mkdir();save(NEW/'manifest.json',manifest)
    if hashlib.sha256(Path(manifest['scenario_files'][0]).read_bytes()).hexdigest()!=manifest['scenario_sha256'][0]:raise ValueError('scene_changed')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LLM_MODEL_NAME=metadata['model_pool'][0],LLM_MODEL_NAMES=','.join(metadata['model_pool']))
    prepare=[]
    for mode in ['off','on']:
        cmd=command.copy()
        for flag in switches:cmd[cmd.index(flag)+1]=mode
        result=subprocess.run(cmd+['--prepare-only'],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
        (NEW/f'prepare_{mode}.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode or (NEW/'temporal_full').exists():raise RuntimeError('temporal_actual_prepare_failed_'+mode)
        prepare.append(dict(mode=mode,exit_code=0,new_API_requests=0,GPU_execution=False))
    save(NEW/'entry_prepare_validation.json',dict(passed=True,records=prepare,not_performance_evidence=True))
    paths=list(json.loads((OLD/'source_sha256.json').read_text()))+[p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py')]
    hashes={}
    for name in sorted(set(paths)):
        source=ROOT/name;hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        dest=NEW/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    save(NEW/'source_sha256.json',hashes)
    metadata.update(command=command,prior_runtime_acceptance=prior,prior_semantic_continuity_solved=False,
        changes='current-velocity-aligned sparse continuation; source-motion facts; between-anchor semantic interaction',
        interaction_radius_m=15.,original_other_semantic_checks_guarded=True,physical_and_background_limits_unchanged=True,
        prototype_regressions_passed=15,planned_attempts=1,new_requests_per_scene_limit=4,
        risk_definition_solver_samples_diffusion_unchanged=True,not_same_model_causal_efficacy_proof=True)
    save(NEW/'launch.json',metadata)
    print(json.dumps(dict(prepared=str(NEW),source_count=len(hashes),actual_prepare_runs=2,new_API_requests=0)))


if __name__=='__main__':main()
