"""冻结旧/新十场配对：只用完全空闲GPU，终态后计算独立验收，不自动重试。"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from model_pool_protocol import inspect_model_source,verify_pool_transition

ROOT=Path('/home2/zhaoyx/scenario-dreamer')


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(path)


def idle_from_rows(gpu_rows,process_rows,reserved=()):
    """瞬时利用率低仍可能有其他CUDA进程；有任何现存计算上下文都继续排队。"""
    occupied={line.split(',')[0].strip() for line in process_rows.splitlines() if line.strip()}
    result=[]
    for line in gpu_rows.splitlines():
        if not line.strip():continue
        index,uuid,util,free=[x.strip() for x in line.split(',')]
        if index not in reserved and uuid not in occupied and int(util)<=10 and int(free)>=20000:result.append(index)
    return result


def idle_gpus(reserved):
    args=['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.free','--format=csv,noheader,nounits']
    gpu=subprocess.run(args,capture_output=True,text=True,check=True).stdout
    actors=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
    return idle_from_rows(gpu,actors,reserved)


def verify_sources(protocol,hashes,root=ROOT):
    bad=[name for name,digest in hashes.items() if name!='policies/llm_adversarial_planner.py'
         and (not (root/name).is_file() or hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest)]
    transition=verify_pool_transition(protocol['model_source'],inspect_model_source(root/'policies/llm_adversarial_planner.py'))
    return dict(verified=not bad and transition['verified'],unexpected_source_changes=bad,model_source=transition)


def child_command(prepared,manifest,output):
    """保留冻结参数，只替换本阶段产物位置；已确认保护后才传递运行标记。"""
    command=list(prepared['launch_command_pending_confirmation'])
    for flag,value in [('--manifest',manifest),('--output-dir',output)]:command[command.index(flag)+1]=str(value)
    command.append('--free-protection-confirmed')
    return ['bash','scripts/run_with_llm_provider.sh','dashscope',*command]


def predecessor_live(directory):
    """核对实际命令行，不能只因旧PID文件存在就认定前置仍在运行。"""
    pid_path=directory.with_suffix('.pid')
    if not pid_path.exists():return False
    try:
        pid=int(pid_path.read_text().strip());os.kill(pid,0)
        state=Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].strip().split()[0]
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        return state!='Z' and directory.name in command
    except (ProcessLookupError,FileNotFoundError,ValueError):return False


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--stage',choices=('development','validation'),required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--development-result',type=Path)
    parser.add_argument('--free-protection-confirmed',action='store_true')
    args=parser.parse_args()
    if not args.free_protection_confirmed:raise ValueError('free_only_protection_confirmation_required_before_launch')
    protocol=json.loads((args.protocol/'protocol.json').read_text())
    hashes=json.loads((args.protocol/'source_sha256.json').read_text())
    prepared=json.loads((args.protocol/'prepared_arms.json').read_text())
    check=verify_sources(protocol,hashes)
    if not check['verified']:raise RuntimeError('source_drift_before_launch')
    if args.stage=='validation':
        if args.development_result is None:raise ValueError('validation requires a completed development result')
        prior=args.development_result
        if predecessor_live(prior):raise ValueError('development_process_still_live')
        if not (prior/'launcher_result.json').exists():raise ValueError('development terminal result missing')
        result=json.loads((prior/'launcher_result.json').read_text())
        if result.get('status')!='completed' or not result.get('acceptance_passed'):raise ValueError('development_gate_not_passed')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    main.created_output=out
    out.with_suffix('.pid').write_text(str(os.getpid()))
    manifest=json.loads((args.protocol/f'{args.stage}10.json').read_text())
    save(out/'manifest.json',manifest);save(out/'source_sha256.json',hashes)
    shutil.copytree(args.protocol/'source',out/'source_snapshot')
    arms=list(prepared);pool=protocol['frozen_model_pool']
    save(out/'launch.json',dict(arms=arms,stage=args.stage,protocol=str(args.protocol),models=pool,
        max_frames=400,max_api_calls_per_scene=4,planned_attempts=20,free_protection_operator_confirmed=True,
        source_model_pool=protocol['model_source'],started_at=time.time()))
    checks=[];records=[];pending=list(arms);active={};reserved=set();source_failed=False
    def receipt(arm,phase):
        nonlocal source_failed
        value=dict(arm=arm,phase=phase,checked_at=time.time(),**verify_sources(protocol,hashes))
        checks.append(value);save(out/'runtime_source_integrity.json',dict(checks=checks))
        source_failed|=not value['verified'];return value['verified']
    with ThreadPoolExecutor(max_workers=2) as executor:
        while pending or active:
            available=idle_gpus(reserved) if pending and not source_failed else []
            while pending and available and not source_failed and len(active)<2:
                arm=pending.pop(0);gpu=available.pop(0)
                if not receipt(arm,'before'):
                    pending.insert(0,arm);break
                command=child_command(prepared[arm],out/'manifest.json',out/arm)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',LLM_MODEL_NAME=pool[0],LLM_MODEL_NAMES=','.join(pool),
                    PYTHONPATH='.:tests:safe-sim:safe-sim/trajdata/src')
                log=(out/f'{arm}.log').open('xb')
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                save(out/f'{arm}_pid.json',dict(pid=process.pid,gpu=gpu,started_at=time.time(),command=command))
                reserved.add(gpu);future=executor.submit(process.wait);active[future]=(arm,gpu,process,log)
            if not active:
                if source_failed:break
                save(out/'queue_status.json',dict(state='waiting_for_completely_idle_gpu',pending=pending,checked_at=time.time()))
                time.sleep(90);continue
            # 超时仅结束一次观察，不代表子进程结束，绝不能据此重启或重复运行。
            done,_=wait(active,timeout=5400,return_when=FIRST_COMPLETED)
            for future in done:
                arm,gpu,process,log=active.pop(future);log.close();reserved.remove(gpu)
                receipt(arm,'after');records.append(dict(arm=arm,exit_code=future.result(),finished_at=time.time()))
                save(out/'launcher_result.partial.json',dict(groups=records,pending=pending,
                    active=[x[0] for x in active.values()],source_failed=source_failed))
    from pool_acceptance import audit_pool
    audit=audit_pool(out);audit_path=ROOT/'work'/f'{out.name}_acceptance.json';save(audit_path,audit)
    status='completed' if not pending and not source_failed and all(x['exit_code']==0 for x in records) else 'completed_with_group_failures'
    save(out/'launcher_result.json',dict(status=status,groups=records,unstarted_groups=pending,
        acceptance_passed=audit['development_gate']['passed'],acceptance_path=str(audit_path),finished_at=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        # 框架异常不清理历史、不自动重跑其他组；只保存不含异常正文的终态标记。
        output=getattr(main,'created_output',None)
        if output is not None:
            save(output/'launcher_result.json',dict(status='launcher_failure',error_type=type(error).__name__,finished_at=time.time()))
        raise
