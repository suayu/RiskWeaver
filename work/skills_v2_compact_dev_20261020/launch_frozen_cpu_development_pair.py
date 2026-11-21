"""固定v2双组开发验证：旧路径可回退，CPU资源独立，所有预定场景保留。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
NAME='skills_v2_frozen_development10_cpu_pair_20261020_r1'
OUT=ROOT/'experiments'/NAME
PYTHON='/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'
DEV=ROOT/'work/skills_v2_compact_dev_20261020'


def dump(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=False)
    (ROOT/'experiments'/f'{NAME}.pid').write_text(str(os.getpid())+'\n')
    # 必须使用先前同一十场，不能看过结果后移除失败或无交互场景。
    previous=ROOT/'experiments/skills_stage_d_development10_bailian_20261020_r2/manifest.json'
    manifest=json.loads(previous.read_text())
    if manifest['original_scenario_indices']!=[53,56,74,80,103,140,181,182,207,220]:
        raise ValueError('development manifest differs from fixed protocol')
    dump('manifest.json',manifest)
    hashes=json.loads((ROOT/'work/skills_validation_protocol_20261020_v1_waitfix/source_sha256.json').read_text())
    if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):
        raise RuntimeError('v1 source drift before v2 freeze')
    for p in DEV.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.json'}:
            hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    hashes[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump('source_sha256.json',hashes)
    # 精确源码快照用于审阅和后续复现；权重/环境/凭据均不复制。
    snapshot=OUT/'source_snapshot'
    for name,digest in hashes.items():
        p=ROOT/name;q=snapshot/name;q.parent.mkdir(parents=True,exist_ok=True)
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('source changed while freezing')
        q.write_bytes(data)
    specs=[('legacy', 'off','off','off','off'),('skills_v2','on','on','on','on')]
    dump('launch.json',dict(version='skills-v2-frozen-cpu-development-20261020',
        arms=[s[0] for s in specs],provider='dashscope',model='qwen3.8-flash',profile=False,dynamic_only=True,
        device='cpu',threads_per_worker=2,max_parallel=2,max_api_calls_per_scene=4,
        scenario_count_per_arm=10,seed=42,max_frames=400,max_seconds_per_arm=43200,
        solver=dict(inner_lr=.2,inner_beta=.5,n_guide_steps=2),weights=[1.5,1,2,4],attack_duration_frames=10,
        analysis_scope='paired CPU bundle comparison; cannot attribute all changes to skill text; GPU v1 matrix unchanged',
        selection_scope='same exposed development ten, not held-out evidence',started_at=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        LLM_MODEL_NAME='qwen3.8-flash',LLM_MODEL_NAMES='qwen3.8-flash',PYTHONUNBUFFERED='1',
        PYTHONPATH='.:tests:safe-sim:safe-sim/trajdata/src:'+str(DEV))
    stop=OUT/'planner_service_stop.json'
    def run(spec):
        arm,skills,compact,event,preflight=spec
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):
            return dict(arm=arm,exit_code=1,failure='source_drift_before_launch')
        cmd=['nice','-n','10','bash','scripts/run_with_llm_provider.sh','dashscope',PYTHON,
            str(DEV/'run_closed_loop.py'),'--device','cpu','--skills',skills,'--candidate-expansion',skills,
            '--skill-level','full','--compact-view',compact,'--event-budget',event,'--reachability-preflight',preflight,
            '--group','riskweaver_profile_off','--manifest',str(OUT/'manifest.json'),'--output-dir',str(OUT/arm),
            '--model','qwen3.8-flash','--models','qwen3.8-flash','--policy','idm','--seed','42','--steps','400',
            '--stop-file',str(stop)]
        with (OUT/f'{arm}.log').open('w') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            dump(f'{arm}_pid.json',dict(pid=p.pid,started_at=time.time()))
            timed_out=False
            try:code=p.wait(timeout=43200)
            except subprocess.TimeoutExpired:
                timed_out=True;p.terminate()
                try:code=p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();code=p.wait()
        return dict(arm=arm,exit_code=code,timed_out=timed_out)
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,s) for s in specs]):
            results.append(future.result());dump('launcher_result.partial.json',dict(groups=results))
    changed=[n for n,h in hashes.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h]
    dump('launcher_result.json',dict(groups=results,
        status='completed' if all(r['exit_code']==0 for r in results) and not changed else 'completed_with_failures',
        source_changes=changed,finished_at=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():dump('launcher_result.json',dict(status='launcher_failure',error_type=type(error).__name__,finished_at=time.time()))
        raise
