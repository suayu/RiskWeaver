"""一个自己的共享GPU工作进程；17GiB准入/14GiB上限/3GiB预留，失败不重启。"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from launch_pool_pair import verify_sources


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def select_shared_gpu(query,allowed):
    """共享准入只依显存围栏和已批准机型；繁忙利用率不再误套为空闲GPU要求。"""
    available=[]
    for line in query.splitlines():
        index,name,free,util=[v.strip() for v in line.split(',')]
        if index in allowed and 'V100S' in name and int(free)>=17*1024:available.append((index,int(free),int(util)))
    return max(available,key=lambda r:r[1])[0] if available else None


def worker_environment(protocol,gpu):
    """显式池优先于llm.env中的旧默认模型，避免原物理预算入口一致性校验失败。"""
    pool=protocol['frozen_model_pool']
    return dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',PYTHONUNBUFFERED='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LLM_MODEL_NAME=pool[0],LLM_MODEL_NAMES=','.join(pool))


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.protocol/'protocol.json').read_text());hashes=json.loads((a.protocol/'source_sha256.json').read_text())
    if not verify_sources(protocol,hashes)['verified']:raise ValueError('source_changed_before_queue')
    out=Path(protocol['worker_output']).parent;out.mkdir(exist_ok=False);out.with_suffix('.pid').write_text(str(os.getpid()))
    for name in ['protocol.json','source_sha256.json','manifest.json']:shutil.copyfile(a.protocol/name,out/name)
    shutil.copytree(a.protocol/'source',out/'source_snapshot')
    save(out/'launch.json',dict(controller_pid=os.getpid(),planned_scenes=[74],planned_attempts=1,started_at=time.time(),
        shared_GPU_authorized=True,own_cap_gib=14,minimum_free_gib=17,maximum_own_workers=1,new_diffusion_refinement=False))
    try:
        while True:
            if shutil.disk_usage(ROOT).free<5.5*2**30:raise RuntimeError('insufficient_disk_space')
            check=verify_sources(protocol,hashes);save(out/'runtime_source_integrity.json',check)
            if not check['verified']:raise ValueError('algorithm_drift_in_queue')
            query=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
            gpu=select_shared_gpu(query,protocol['allowed_GPU_indices'])
            if gpu is not None:break
            save(out/'queue_status.json',dict(state='waiting_shared_V100S_17GiB',checked_at=time.time(),allowed_GPU_indices=protocol['allowed_GPU_indices']))
            time.sleep(90)
        command=['bash','scripts/run_with_llm_provider.sh','dashscope',*protocol['command']]
        env=worker_environment(protocol,gpu)
        with (out/'temporal_local.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(out/'method_pid.json',dict(pid=child.pid,gpu=gpu,command=command,started_at=time.time()));code=child.wait()
        check=verify_sources(protocol,hashes);save(out/'runtime_source_integrity.json',check)
        save(out/'launcher_result.json',dict(status='completed' if code==0 and check['verified'] else 'failed',exit_code=code,
            source_verified=check['verified'],finished_at=time.time(),automatic_retry=False,goal_complete=False))
    except Exception as error:
        save(out/'launcher_result.json',dict(status='failed',error_type=type(error).__name__,finished_at=time.time(),automatic_retry=False,goal_complete=False));raise


if __name__=='__main__':main()
