"""一次性CPU诊断监督器：记录真实退出码，失败不重试，不影响已冻结开发队列。"""
from pathlib import Path
import json,os,subprocess,sys,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
NAME='anchor_objective_cpu_scene074_20261030_r1'


def main():
    out=ROOT/'experiments'/NAME;log=ROOT/'work'/(NAME+'_worker.log')
    if out.exists() or log.exists():raise FileExistsError('preserve_previous_anchor_probe')
    command=[sys.executable,str(ROOT/'work/anchor_refinement_20261030/probe.py'),'--output',str(out),
        '--seeds','42','--modes','legacy_dense_huber','dense_squared','sparse_squared']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    with log.open('x') as file:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT)
        with (ROOT/'work'/(NAME+'_worker_pid.json')).open('x') as f:json.dump(dict(pid=child.pid,command=command,started_at=time.time()),f,indent=2)
        code=child.wait()
    record=dict(exit_code=code,status='completed' if code==0 else 'failed',finished_at=time.time(),new_API_requests=0,new_execution_frames=0)
    with (ROOT/'work'/(NAME+'_supervisor_result.json')).open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(record))


if __name__=='__main__':main()
