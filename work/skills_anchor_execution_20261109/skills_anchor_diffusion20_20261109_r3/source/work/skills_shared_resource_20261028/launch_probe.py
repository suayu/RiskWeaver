"""共享资源探测控制器：只停止自身探测子进程，绝不影响其他用户或旧队列。"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from launch_pool_pair import verify_sources
from resource_policy import eligible_shared


def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(p)


def gpu_rows():
    return subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)


def stop_owned(child,marker):
    if child.poll() is not None:return
    p=Path('/proc')/str(child.pid)/'cmdline';command=p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else ''
    if 'skills_shared_resource_20261028/run_capped.py' not in command or marker not in command:
        raise RuntimeError('refuse_to_signal_unverified_process')
    child.terminate()
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();protocol=json.loads((args.protocol/'protocol.json').read_text());hashes=json.loads((args.protocol/'source_sha256.json').read_text())
    if not verify_sources(protocol,hashes)['verified']:raise ValueError('resource_probe_source_drift')
    out=args.output;out.mkdir(exist_ok=False);out.with_suffix('.pid').write_text(str(os.getpid()))
    for name in ['manifest.json','protocol.json','source_sha256.json']:shutil.copyfile(args.protocol/name,out/name)
    shutil.copytree(args.protocol/'source',out/'source_snapshot')
    save(out/'launch.json',dict(controller_pid=os.getpid(),user_shared_probe_authorized=True,not_formal_comparison=True,started_at=time.time()))
    while True:
        candidates=eligible_shared(gpu_rows(),protocol['cap_gib'],protocol['reserve_gib'],protocol['allowed_GPU_indices'])
        if candidates:break
        save(out/'queue_status.json',dict(state='waiting_for_shared_memory_headroom',checked_at=time.time(),minimum_free_gib=17))
        time.sleep(90)
    gpu=candidates[0]['index'];command=list(protocol['command'])
    command[command.index('--output-dir')+1]=str(out/'probe')
    command=['bash','scripts/run_with_llm_provider.sh','dashscope',*command,'--free-protection-confirmed']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        LLM_MODEL_NAME=protocol['frozen_model_pool'][0],LLM_MODEL_NAMES=','.join(protocol['frozen_model_pool']),
        PYTHONPATH='.:tests:safe-sim:safe-sim/trajdata/src')
    reason=None;start=time.monotonic()
    with (out/'probe.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'probe_pid.json',dict(pid=child.pid,gpu=gpu,command=command,admission=candidates[0],started_at=time.time()))
        while child.poll() is None:
            if time.monotonic()-start>protocol['max_wall_seconds']:
                reason='probe_wall_budget_exhausted';stop_owned(child,str(out));break
            rows=gpu_rows();selected=next(l.split(',') for l in rows.splitlines() if l.split(',')[0].strip()==gpu)
            free=float(selected[2].strip())/1024
            with (out/'gpu_observations.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(time=time.time(),gpu=gpu,free_gib=free,util=float(selected[3].strip())))+'\n')
            if free<protocol['watchdog_min_free_gib']:
                reason='shared_GPU_low_free_memory_watchdog';stop_owned(child,str(out));break
            time.sleep(protocol['watchdog_poll_seconds'])
    from probe_acceptance import assess
    acceptance=assess(out,child.returncode,reason)
    save(out/'resource_acceptance.json',acceptance)
    save(out/'launcher_result.json',dict(status='resource_probe_passed' if acceptance['passed'] else 'resource_probe_failed_or_inconclusive',
        exit_code=child.returncode,stop_reason=reason,resource_gate_passed=acceptance['passed'],finished_at=time.time(),not_performance_evidence=True))
    print(json.dumps(acceptance,ensure_ascii=False))


if __name__=='__main__':main()
