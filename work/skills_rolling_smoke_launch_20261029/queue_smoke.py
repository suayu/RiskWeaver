"""单场共享GPU队列：只允许GPU0/17GiB准入，不迁移或重启运行任务。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
RUN=ROOT/'experiments/skills_rolling_future_scene074_20261029_r2'
sys.path.insert(0,str(ROOT/'work/skills_shared_resource_20261028'))
from resource_policy import eligible_shared


def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)


def main():
    launch=json.loads((RUN/'launch.json').read_text());hashes=json.loads((RUN/'source_sha256.json').read_text())
    if not json.loads((RUN/'entry_prepare_validation.json').read_text()).get('passed'):raise ValueError('prepare_gate_not_passed')
    if (RUN/'method_pid.json').exists() or (RUN/'launcher_result.json').exists():raise FileExistsError('never_restart_prior_attempt')
    # 本次排队期间任何源码改变均停止派发；运行期额度注释由原公共验证器处理。
    while True:
        changed=[p for p,h in hashes.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        if changed:raise ValueError('source_changed_before_dispatch:'+','.join(changed))
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
        eligible=eligible_shared(rows,14.,3.,allowed=('0',))
        if eligible:break
        save(RUN/'queue_status.json',dict(controller_pid=os.getpid(),status='waiting_for_GPU0_17GiB_headroom',checked_at=time.time()))
        time.sleep(90)
    save(RUN/'runtime_source_integrity.json',dict(verified=True,source_count=len(hashes),checked_at=time.time()))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        LLM_MODEL_NAME=launch['model_pool'][0],LLM_MODEL_NAMES=','.join(launch['model_pool']))
    with (RUN/'run.log').open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(launch['command'],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save(RUN/'method_pid.json',dict(pid=child.pid,gpu='0',command=launch['command'],started_at=time.time()))
        code=child.wait()
    save(RUN/'launcher_result.json',dict(status='completed' if code==0 else 'failed',exit_code=code,planned_attempts=1,
        finished_at=time.time(),closed_loop_acceptance_pending=True,goal_complete=False))


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(RUN/'launcher_result.json',dict(status='controller_failure',error_type=type(e).__name__,reason=str(e),goal_complete=False))
        raise
