"""两组独立顺序实验；共享V100S显存围栏沿用已批准配置，不改算法/不自动重试。"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'work/skills_modelpool_budget_dev_20261022'),str(ROOT/'work/skills_candidate_local_runtime_20261030')]
from launch_pool_pair import verify_sources
from queue_smoke import select_shared_gpu


def save(path,value):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def select_rag_gpu(query):
    """队列额外留1GiB给CUDA初始化；工作进程14GiB上限与3GiB预留不降低。"""
    lines=[line for line in query.splitlines() if int(line.split(',')[2].strip())>=18*1024]
    return select_shared_gpu('\n'.join(lines),['0','1','2','3'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.protocol/'protocol.json').read_text());hashes=json.loads((a.protocol/'source_sha256.json').read_text())
    if protocol['arms']!=['skills_no_rag','skills_rag']:raise ValueError('only_two_RAG_arms_allowed')
    if not verify_sources(protocol,hashes)['verified']:raise ValueError('source_changed_before_RAG_run')
    root=Path(protocol['experiment_root']);root.mkdir(exist_ok=False);root.with_suffix('.pid').write_text(str(os.getpid()))
    for name in ['protocol.json','source_sha256.json','manifest.json']:shutil.copyfile(a.protocol/name,root/name)
    shutil.copytree(a.protocol/'source',root/'source_snapshot');results={}
    try:
        for arm in protocol['arms']:
            while True:
                if shutil.disk_usage(ROOT).free<5.5*2**30:raise RuntimeError('insufficient_disk_space')
                check=verify_sources(protocol,hashes);save(root/'runtime_source_integrity.json',check)
                if not check['verified']:raise ValueError('algorithm_changed')
                seed=protocol.get('inherited_seed')
                if seed:
                    import hashlib
                    if hashlib.sha256(Path(seed).read_bytes()).hexdigest()!=protocol['inherited_seed_sha256']:raise ValueError('frozen_inherit_seed_changed')
                query=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
                gpu=select_rag_gpu(query)
                if gpu is not None:break
                save(root/'queue_status.json',dict(pending_arm=arm,status='waiting_shared_18GiB_including_CUDA_startup_headroom',checked_at=time.time()));time.sleep(90)
            models=protocol['models'];env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',PYTHONUNBUFFERED='1',
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',LLM_MODEL_NAME=models[0],LLM_MODEL_NAMES=','.join(models))
            command=['bash','scripts/run_with_llm_provider.sh','dashscope',*protocol['commands'][arm]]
            with (root/(arm+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                save(root/(arm+'_pid.json'),dict(pid=child.pid,gpu=gpu,command=command));code=child.wait()
            results[arm]=dict(exit_code=code,finished_at=time.time());save(root/'launcher_result.partial.json',results)
            # 最终服务/运行故障不继续消耗另一组请求；保留未尝试分母，不拼接重试。
            if code:raise RuntimeError(arm+'_failed')
        check=verify_sources(protocol,hashes);save(root/'runtime_source_integrity.json',check)
        if not check['verified']:raise ValueError('source_changed_during_RAG_run')
        save(root/'launcher_result.json',dict(status='completed',groups=results,planned_attempts=protocol['planned_attempts'],goal_complete=False))
    except Exception as error:
        save(root/'launcher_result.json',dict(status='failed',groups=results,error_type=type(error).__name__,planned_attempts=protocol['planned_attempts'],goal_complete=False));raise


if __name__=='__main__':main()
