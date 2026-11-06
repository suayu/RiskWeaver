"""同一已记录计划的完整单场旧/新检验，不创建真实LLM客户端。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
NAME='skills_joint_recovery_fixed_intent_scene056_full_20261020_r2'
OUT=ROOT/'experiments'/NAME
PYTHON='/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'


def dump(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))


def main():
    OUT.mkdir(exist_ok=False);(ROOT/'experiments'/f'{NAME}.pid').write_text(str(os.getpid()))
    manifest=json.loads((ROOT/'experiments/skills_v2_frozen_development10_cpu_pair_20261020_r1/manifest.json').read_text())
    index=manifest['original_scenario_indices'].index(56)
    manifest.update(scenario_count=1,original_scenario_indices=[56],scenario_indices=[56],
                    scenario_files=[manifest['scenario_files'][index]],scenario_sha256=[manifest['scenario_sha256'][index]])
    dump('manifest.json',manifest)
    folder=ROOT/'work/skills_v3_candidate_recovery_fast_20261020'
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.py'))}
    hashes['work/skills_lifecycle_dev_20261020/fixed_intent_lifecycle.py']=hashlib.sha256((ROOT/'work/skills_lifecycle_dev_20261020/fixed_intent_lifecycle.py').read_bytes()).hexdigest()
    dump('source_sha256.json',hashes)
    dump('launch.json',dict(arms=['recovery_off','recovery_on'],scene=56,seed=42,max_frames=400,inference_calls=0,
        weights=[1.5,1,2,4],solver=dict(inner_lr=.2,inner_beta=.5,n_guide_steps=2),
        device='cpu',recovery_acceptance_budget_seconds=30.,scope='fixed recorded intent; full-episode execution, not fresh LLM generation coverage',started_at=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             PYTHONUNBUFFERED='1',PYTHONPATH='.:tests:safe-sim:safe-sim/trajdata/src:work/skills_v3_candidate_recovery_fast_20261020')
    results=[]
    for option in ['off','on']:
        arm=f'recovery_{option}'
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):raise RuntimeError('source_drift')
        command=[PYTHON,str(folder/'run_fixed_recovery.py'),'--joint-candidate-recovery',option,
            '--group','riskweaver_profile_off','--manifest',str(OUT/'manifest.json'),'--output-dir',str(OUT/arm),
            '--model','fixed_logged_intent','--models','fixed_logged_intent','--policy','idm','--seed','42','--steps','400']
        with (OUT/f'{arm}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            dump(f'{arm}_pid.json',dict(pid=child.pid,started_at=time.time()))
            timeout=False
            try:code=child.wait(timeout=5400)
            except subprocess.TimeoutExpired:
                timeout=True;child.terminate()
                try:code=child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();code=child.wait()
        results.append(dict(arm=arm,exit_code=code,timed_out=timeout))
        dump('launcher_result.partial.json',dict(groups=results))
    dump('launcher_result.json',dict(groups=results,status='completed' if all(r['exit_code']==0 for r in results) else 'completed_with_failures',finished_at=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():dump('launcher_result.json',dict(status='launcher_failure',error_type=type(error).__name__))
        raise
