"""在空闲GPU0顺序比较同一已记录计划的1秒/3秒控制，严格禁止新LLM请求。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
NAME='skills_fixed_intent_lifecycle_scene182_20261020_r1'
OUT=ROOT/'experiments'/NAME
PYTHON='/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'


def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def idle_gpu0():
    rows=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()
    entry=next(r.split(',') for r in rows if r.split(',')[0].strip()=='0')
    actors=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines()
    return int(entry[2])>=14000 and int(entry[3])<=10 and entry[1].strip() not in [a.strip() for a in actors]


def main():
    os.chdir(ROOT);OUT.mkdir(exist_ok=False)
    (ROOT/'experiments'/f'{NAME}.pid').write_text(str(os.getpid())+'\n')
    parent=ROOT/'experiments/skills_stage_d_paired_scene182_bailian_20261020_r1'
    manifest=json.loads((parent/'manifest.json').read_text())
    if manifest['original_scenario_indices']!=[182]:raise ValueError('fixed scene mismatch')
    dump('manifest.json',manifest)
    # 模型池注释因免费额度耗尽发生变化；只记录真实当前源码，不恢复已停用模型。
    known=json.loads((ROOT/'work/skills_validation_protocol_20261020_v1_waitfix/source_sha256.json').read_text())
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in known}
    for name in ['fixed_intent_lifecycle.py','test_fixed_intent_lifecycle.py','launch_fixed_lifecycle.py']:
        f=ROOT/'work/skills_lifecycle_dev_20261020'/name
        hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    dump('source_sha256.json',hashes)
    dump('launch.json',dict(arms=['duration10','duration30'],scene=182,seed=42,max_frames=400,
        inference_calls=0,source_plan='previously accepted Bailian qwen3.8-flash intent at scene182 step0',
        scope='fixed-plan lifecycle diagnostic only; not new LLM success, not generation coverage experiment',
        solver=dict(inner_lr=.2,inner_beta=.5,n_guide_steps=2),weights=[1.5,1,2,4],gpu='0',started_at=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',
        PYTHONPATH='.:tests:safe-sim:safe-sim/trajdata/src:work/skills_lifecycle_dev_20261020')
    results=[]
    for duration in [10,30]:
        started=time.monotonic()
        while not idle_gpu0():
            if time.monotonic()-started>1800:raise RuntimeError('idle_gpu_wait_budget_exhausted')
            dump('queue_status.json',dict(state='waiting_for_completely_idle_gpu0',checked_at=time.time()))
            time.sleep(30)
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=d for n,d in hashes.items()):
            raise RuntimeError('fixed_intent_source_drift')
        arm=f'duration{duration}'
        cmd=[PYTHON,'work/skills_lifecycle_dev_20261020/fixed_intent_lifecycle.py','--duration',str(duration),
            '--group','riskweaver_profile_off','--manifest',str(OUT/'manifest.json'),'--output-dir',str(OUT/arm),
            '--model','fixed_logged_intent','--models','fixed_logged_intent','--policy','idm','--seed','42','--steps','400']
        with (OUT/f'{arm}.log').open('w') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            dump(f'{arm}_pid.json',dict(pid=p.pid,gpu='0',started_at=time.time()))
            timeout=False
            try:code=p.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                timeout=True;p.terminate()
                try:code=p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        results.append(dict(arm=arm,exit_code=code,timed_out=timeout))
        dump('launcher_result.partial.json',dict(groups=results))
    dump('launcher_result.json',dict(groups=results,status='completed' if all(r['exit_code']==0 for r in results) else 'completed_with_failures',finished_at=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():dump('launcher_result.json',dict(status='launcher_failure',error_type=type(error).__name__,finished_at=time.time()))
        raise
