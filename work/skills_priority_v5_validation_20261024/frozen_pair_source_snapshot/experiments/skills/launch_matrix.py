"""技能四组配对的固定协议运行器：空闲卡并行、有门槛的分阶段启动。"""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
PY='/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'
ARMS=[('legacy','off','off','none'),('catalog_expanded','on','on','none'),
      ('skills','on','off','full'),('skills_expanded','on','on','full')]


def save(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    temporary.replace(path)


def live(pid):
    try:
        os.kill(int(pid),0)
        # 僵尸进程已结束计算，不应仅因PID暂未回收而永久等待。
        stat=Path(f'/proc/{int(pid)}/stat')
        return not stat.exists() or stat.read_text().rpartition(')')[2].strip().split()[0]!='Z'
    except ProcessLookupError:return False


def prerequisites(path,kind):
    """真实进程结束且终态产物齐全才判门槛；不因观察超时重启任务。"""
    pid_path=path.with_suffix('.pid')
    if pid_path.exists() and live(pid_path.read_text().strip()):return None,'predecessor_live'
    final=path/'launcher_result.json'
    if not final.exists():return False,'predecessor_terminal_result_missing'
    result=json.loads(final.read_text())
    if result.get('status')!='completed':return False,'predecessor_not_completed'
    from experiments.skills.audit import audit_experiment
    audit=audit_experiment(path)
    groups=audit['groups']
    if not audit['all_terminal']:return False,'predecessor_group_not_terminal'
    for name,group in groups.items():
        if not group['artifacts_complete'] or group['runtime_failures'] or group['planner_service_failures'] or group['background_collision_frames']:
            return False,'predecessor_integrity_or_runtime_failure:'+name
    expanded=groups.get('skills_expanded',{})
    required=1 if kind=='smoke' else 6
    if expanded.get('actual_attack_scenes',0)<required:return False,'insufficient_actual_attack_scenes'
    return True,'passed'


def idle_gpus(reserved):
    result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows=[list(map(int,line.split(','))) for line in result.stdout.strip().splitlines()]
    return [str(index) for index,util,free in rows if index in (0,3) and util<=10 and free>=20000 and str(index) not in reserved]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--stage',choices=('development','validation'),required=True)
    parser.add_argument('--wait-for',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.protocol/'protocol.json').read_text())
    manifest=json.loads((args.protocol/(args.stage+'10.json')).read_text())
    source_hashes=json.loads((args.protocol/'source_sha256.json').read_text())
    extras=['experiments/skills/audit.py','experiments/skills/run_instrumented.py','experiments/skills/launch_matrix.py']
    source_hashes.update({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in extras})
    for name,digest in source_hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('source_drift_before_matrix:'+name)
    save(out/'source_sha256.json',source_hashes)
    save(out/'manifest.json',manifest)
    save(out/'launch.json',dict(arms=ARMS,stage=args.stage,model=protocol['model'],provider='dashscope',
        protocol=str(args.protocol),planned_attempts=40,started_at=time.time(),wait_for=str(args.wait_for),
        runtime_note='IDM; profile off; no static obstacles; max400 frames; max4 API; fixed optimizer; all planned scenes counted'))
    while True:
        passed,reason=prerequisites(args.wait_for,'smoke' if args.stage=='development' else 'development')
        save(out/'prerequisite_status.json',dict(passed=passed,reason=reason,checked_at=time.time()))
        if passed is not None:break
        # 等待已知进程退出；这不是反复读取结果，退出后立即执行一次门槛审计。
        try:
            import select
            predecessor_pid=int(args.wait_for.with_suffix('.pid').read_text())
            descriptor=os.pidfd_open(predecessor_pid)
        except (AttributeError,OSError):time.sleep(5400)
        else:
            try:select.select([descriptor],[],[],5400)
            finally:os.close(descriptor)
    if not passed:
        save(out/'launcher_result.json',dict(status='prerequisite_gate_failed',reason=reason,groups=[],finished_at=time.time()));return
    pending=list(ARMS);active={};reserved=set();records=[];stop_new=False
    with ThreadPoolExecutor(max_workers=2) as pool:
        while pending or active:
            available=idle_gpus(reserved) if pending and not stop_new else []
            while available and pending and len(active)<2 and not stop_new:
                name,skills,expansion,level=pending.pop(0);gpu=available.pop(0)
                for source,digest in source_hashes.items():
                    if hashlib.sha256((ROOT/source).read_bytes()).hexdigest()!=digest:
                        raise RuntimeError('source_drift_during_matrix:'+source)
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,
                    PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',PYTHONUNBUFFERED='1',
                    LLM_MODEL_NAME=protocol['model'],LLM_MODEL_NAMES=protocol['model'])
                command=['bash','scripts/run_with_llm_provider.sh','dashscope',PY,'experiments/skills/run_instrumented.py',
                    '--skills',skills,'--candidate-expansion',expansion,'--skill-level',level,
                    '--group','riskweaver_profile_off','--manifest',str(out/'manifest.json'),
                    '--output-dir',str(out/name),'--model',protocol['model'],'--models',protocol['model'],
                    '--policy','idm','--seed','42','--steps','400']
                log=(out/f'{name}.log').open('xb')
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                reserved.add(gpu);save(out/f'{name}_pid.json',dict(pid=process.pid,gpu=gpu,command=command,started_at=time.time()))
                future=pool.submit(process.wait);active[future]=(name,gpu,process,log)
            if not active:
                if stop_new:break
                save(out/'queue_status.json',dict(state='waiting_for_idle_gpu',pending=[a[0] for a in pending],checked_at=time.time()))
                # 这里只排队等待GPU资源，实验结果仍只在进程退出时读取。
                time.sleep(90);continue
            # 阻塞等待子进程退出，不以频繁读取实验产物的方式轮询。
            done,_=wait(active,timeout=5400,return_when=FIRST_COMPLETED)
            for future in done:
                name,gpu,process,log=active.pop(future);log.close();reserved.remove(gpu)
                record=dict(arm=name,exit_code=future.result(),finished_at=time.time());records.append(record)
                result_file=out/name/'results.json';exit_file=out/name/'exit_status.json'
                result=json.loads(result_file.read_text()) if result_file.exists() else {}
                exit_status=json.loads(exit_file.read_text()) if exit_file.exists() else {}
                if result.get('aggregate',{}).get('planner_service_failure_count',0) or exit_status.get('error_type')=='PlannerServiceFailure':
                    stop_new=True
                save(out/'launcher_result.partial.json',dict(groups=records,pending=[a[0] for a in pending],
                    active=[v[0] for v in active.values()],stop_new_for_service_failure=stop_new))
    status='final_planner_service_failure' if stop_new else ('completed_with_group_failures' if any(r['exit_code'] for r in records) else 'completed')
    save(out/'launcher_result.json',dict(status=status,groups=records,
        unstarted_groups=[a[0] for a in pending],finished_at=time.time()))


if __name__=='__main__':
    try:main()
    except Exception:
        # 框架异常保留标准错误与PID；不自动重启或清理其他运行进程。
        raise
