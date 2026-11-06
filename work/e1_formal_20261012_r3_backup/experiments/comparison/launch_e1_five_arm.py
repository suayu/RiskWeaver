"""先五组各一场冒烟，验收通过才运行同一 100 场；按场景分片复用 GPU。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.comparison.e1_protocol import GROUPS,PROFILE_GROUPS,dump_json,select_100,shard_manifest,protocol_snapshot
from experiments.comparison.e1_metrics import aggregate_attempts

PYTHON='/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'


def available_gpus(allowed,min_free_gb=14):
    """只调度显存和利用率满足条件的卡，不终止或修改其他研究任务。"""
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    result=[]
    for row in data.splitlines():
        gpu,free,util=map(int,row.split(','))
        if gpu in allowed and free>=min_free_gb*1024 and util<=10:
            result.append(gpu)
    return result


def smoke_acceptance(directory,group):
    """只验链路、执行和产物，不要求碰撞，不据结果更换冒烟场景。"""
    errors=[]
    result_path=directory/'results.json'
    if not result_path.exists():
        return ['missing_results']
    result=json.loads(result_path.read_text())
    episodes=result.get('episodes',[])
    if len(episodes)!=1:
        return ['wrong_smoke_attempt_count']
    row=episodes[0]
    if row.get('executed_steps',0)<=0:errors.append('no_actual_execution')
    if row.get('generation_failure'):errors.append('generation_failure')
    if row.get('runtime_failure'):errors.append('runtime_failure')
    if row.get('planner_service_failure_count'):errors.append('planner_service_failure')
    if row.get('prediction_service_failure_count'):errors.append('prediction_service_failure')
    if row.get('ea_valid_frame_count',0)<=0:errors.append('no_valid_EA')
    if group in PROFILE_GROUPS or group=='strive':
        if row.get('attack_plan_count',0)<=0:errors.append('no_attack_plan')
        if row.get('attack_executed_frames',0)<=0:errors.append('no_attack_execution')
    if group=='safe_sim' and row.get('attack_executed_frames',0)<=0:
        errors.append('official_guidance_not_executed')
    f=row.get('feasibility',{})
    if f.get('background_collision_frames') or f.get('background_static_collision_frames'):
        errors.append('background_collision')
    if row.get('obstacle_plan_count'):errors.append('unexpected_obstacle_plan')
    trace_path=Path(f.get('execution_trace_path',''))
    if not trace_path.is_file():
        errors.append('missing_trace')
    else:
        try:
            lines=[json.loads(s) for s in trace_path.read_text().splitlines()]
            states=[v for v in lines if v.get('kind')=='state']
            if len(states)!=row['executed_steps']+1:errors.append('state_trace_count_mismatch')
            if any(v.get('static_obstacles')!=[] for v in states):errors.append('unexpected_static_obstacles')
            if group=='safe_sim':
                predictions=[v.get('joint') or {} for v in lines if v.get('kind')=='execution_prediction']
                if not any('safe_sim_causecollision' in p.get('metadata',{}).get('guidance_functions',[]) for p in predictions):
                    errors.append('wrong_safe_sim_guidance')
        except Exception:
            errors.append('invalid_trace_json')
    video=Path(row.get('video_path') or '')
    if not video.is_file():
        errors.append('missing_video')
    else:
        try:
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                'stream=nb_frames','-of','json',str(video)],text=True))
            frames=int(probe['streams'][0]['nb_frames'])
            if frames<row['executed_steps']+1:errors.append('video_missing_terminal_frame')
        except Exception:
            errors.append('video_probe_failure')
    return errors


def job_command(group,manifest,directory,args):
    """使用无密钥命令参数；提供方脚本从用户受限配置加载认证。"""
    command=[PYTHON,'experiments/comparison/run_e1_group.py','--group',group,'--manifest',str(manifest),
        '--output-dir',str(directory),'--policy',args.policy,'--seed',str(args.seed),
        '--model',args.model,'--stop-file',str(args.output_dir/'STOP_LLM_SERVICE.json')]
    return ['bash','scripts/run_with_llm_provider.sh','dashscope',*command] if group in PROFILE_GROUPS else command


def execute_phase(jobs,args,phase):
    """任务队列保留进度和 PID；仅本任务新分片失败时停止后续调度。"""
    pending=list(jobs);running={};finished=[];failed=False
    while pending or running:
        for gpu,(process,job,log,start) in list(running.items()):
            code=process.poll()
            if code is None:
                continue
            log.close()
            errors=smoke_acceptance(job['directory'],job['group']) if phase=='smoke' and code==0 else []
            status={**{k:str(v) if isinstance(v,Path) else v for k,v in job.items()},
                'gpu':gpu,'exit_code':code,'started_at':start,'finished_at':time.time(),
                'acceptance_errors':errors,'passed':code==0 and not errors}
            dump_json(job['directory'].parent/(job['directory'].name+'_launcher_status.json'),status)
            finished.append(status);del running[gpu]
            failed |= not status['passed']
        if (args.output_dir/'STOP_LLM_SERVICE.json').exists():failed=True
        if failed and not running:
            break
        if not failed and pending:
            for gpu in available_gpus(args.gpus,args.min_free_gb):
                if gpu in running or not pending:continue
                job=pending.pop(0)
                env=os.environ.copy()
                env['CUDA_VISIBLE_DEVICES']=str(gpu)
                env['PYTHONPATH']='.:safe-sim:safe-sim/trajdata/src'
                env['PYTHONUNBUFFERED']='1'
                if job['group'] in PROFILE_GROUPS:
                    env['LLM_MODEL_NAME']=args.model;env['LLM_MODEL_NAMES']=args.model
                job['directory'].parent.mkdir(parents=True,exist_ok=True)
                log_path=job['directory'].parent/(job['directory'].name+'.log')
                log=log_path.open('x',encoding='utf-8')
                command=job_command(job['group'],job['manifest'],job['directory'],args)
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                start=time.time()
                running[gpu]=(process,job,log,start)
                dump_json(job['directory'].parent/(job['directory'].name+'_pid.json'),{'pid':process.pid,'gpu':gpu,'command':command})
        dump_json(args.output_dir/'launcher_result.partial.json',{'phase':phase,'finished':finished,
            'running':[{'group':j['group'],'directory':str(j['directory']),'pid':p.pid,'gpu':g} for g,(p,j,l,s) in running.items()],
            'pending_count':len(pending),'failed':failed,'updated_at':time.time()})
        if pending or running:
            time.sleep(5 if running else 30)
    return {'phase':phase,'jobs':finished,'unstarted_jobs':[str(j['directory']) for j in pending],
            'passed':not failed and not pending and len(finished)==len(jobs)}


def merge_formal(jobs,manifest,args):
    """合并时严格验证全局编号、场景来源及五组完整性；不丢弃失败尝试。"""
    results={};expected=set(manifest['original_scenario_indices'])
    for group in GROUPS:
        rows=[]
        for job in jobs:
            if job['group']!=group:continue
            p=job['directory']/'results.json'
            if p.exists():rows.extend(json.loads(p.read_text()).get('episodes',[]))
        ids=[r['scenario_index'] for r in rows]
        if len(ids)!=len(set(ids)):
            raise ValueError('duplicate global scene indices')
        rows.sort(key=lambda r:r['scenario_index'])
        by_id=dict(zip(manifest['original_scenario_indices'],manifest['scenario_files']))
        if any(r['scenario_index'] not in expected or r['scenario_source']!=by_id[r['scenario_index']] for r in rows):
            raise ValueError('global scene source mismatch')
        results[group]={'aggregate':aggregate_attempts(rows,100),'complete':set(ids)==expected,'episodes':rows}
    dump_json(args.output_dir/'comparison_results.json',results)
    from experiments.comparison.e1_reporting import write_tables
    write_tables(results,args.output_dir)
    return all(r['complete'] for r in results.values())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--policy',choices=('idm','rl'),default='idm')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--gpus',default='0,1,2,3')
    parser.add_argument('--min-free-gb',type=float,default=14)
    parser.add_argument('--shard-size',type=int,default=10)
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--smoke-groups',default=','.join(GROUPS))
    args=parser.parse_args()
    args.gpus=[int(x) for x in args.gpus.split(',')]
    smoke_groups=args.smoke_groups.split(',')
    if not smoke_groups or len(set(smoke_groups))!=len(smoke_groups) or any(g not in GROUPS for g in smoke_groups):
        raise ValueError('invalid smoke group selection')
    if args.shard_size<1 or not args.gpus:
        raise ValueError('invalid scheduler configuration')
    # 在创建实验目录前拒绝场景/权重变更，不下载或替换已有权重。
    from experiments.comparison.run_passive_baseline import validate_manifest
    source,_=validate_manifest(args.manifest)
    checkpoint=ROOT/'checkpoints/strive_official_20261008/traffic_model.pth'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!='f977360762b7c2f8d66745fb781b1d1a66c4963a5d48a47bc2ca7646861088cf':
        raise ValueError('STRIVE checkpoint hash changed')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'launcher.pid').write_text(str(os.getpid()),encoding='utf-8')
    dump_json(args.output_dir/'protocol.json',protocol_snapshot(args.model,args.policy,args.seed))
    selected=select_100(source,args.seed)
    manifest100=args.output_dir/'frozen_100_scenes.json'
    dump_json(manifest100,selected)
    smoke_manifest=args.output_dir/'smoke_one_scene.json'
    dump_json(smoke_manifest,shard_manifest(source,[0]))
    source_files=['experiments/comparison/e1_metrics.py','experiments/comparison/e1_protocol.py',
        'experiments/comparison/run_e1_group.py','experiments/comparison/launch_e1_five_arm.py',
        'experiments/comparison/e1_reporting.py',
        'policies/safe_sim_official_guidance.py','policies/diffusion_model_wrapper.py','policies/llm_adversarial_planner.py',
        'policies/ego_profile.py','policies/strive_adapter.py','scenario_generator.py','simulator.py']
    dump_json(args.output_dir/'launch.json',{'argv':sys.argv,'started_at':time.time(),
        'manifest_sha256':hashlib.sha256(manifest100.read_bytes()).hexdigest(),
        'source_manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files}})
    smoke_jobs=[{'group':g,'manifest':smoke_manifest,'directory':args.output_dir/'smoke'/g} for g in smoke_groups]
    smoke=execute_phase(smoke_jobs,args,'smoke')
    dump_json(args.output_dir/'smoke_acceptance.json',smoke)
    if not smoke['passed'] or args.smoke_only:
        dump_json(args.output_dir/'launcher_result.json',{'smoke':smoke,'formal_started':False,'finished_at':time.time()})
        return 0 if smoke['passed'] else 71
    jobs=[]
    for shard,start in enumerate(range(0,100,args.shard_size)):
        m=args.output_dir/'manifests'/f'shard_{shard:03d}.json'
        dump_json(m,shard_manifest(selected,list(range(start,min(start+args.shard_size,100)))))
        for group in GROUPS:
            jobs.append({'group':group,'manifest':m,'directory':args.output_dir/'formal'/group/f'shard_{shard:03d}'})
    formal=execute_phase(jobs,args,'formal')
    complete=merge_formal(jobs,selected,args)
    dump_json(args.output_dir/'launcher_result.json',{'smoke':smoke,'formal':formal,'formal_started':True,
        'all_groups_complete':complete,'finished_at':time.time()})
    return 0 if formal['passed'] and complete else 72


if __name__=='__main__':
    raise SystemExit(main())
