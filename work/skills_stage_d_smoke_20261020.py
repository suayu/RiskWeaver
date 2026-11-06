"""三组同一开发场景闭环；空闲GPU顺序运行，保留每组全部失败与产物。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
OUT=ROOT/'experiments/skills_stage_d_paired_scene182_bailian_20261020_r1'
PY='/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'
ARMS=[('legacy','off','off'),('skills','on','off'),('skills_expanded','on','on')]


def save(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    OUT.mkdir(exist_ok=False)
    manifest=json.loads((ROOT/'experiments/e1_five_arm_idm_10_modelpool_20261014_r1/formal/riskweaver_profile_off/shard_007/manifest.json').read_text())
    manifest.update(comparison_scene_count=1,selection_basis='fixed_development_scene182_chain_diagnostic')
    save('manifest.json',manifest)
    save('launch.json',dict(arms=ARMS,model='qwen3.8-flash',provider='dashscope',planned_attempts=3,
         profile=False,static=False,seed=42,max_frames=400,max_planning_calls=4,started_at=time.time()))
    records=[]
    for name,skills,expansion in ARMS:
        # 不在繁忙卡上共享；三个方法使用相同物理GPU顺序执行。
        while True:
            result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            rows=[list(map(int,line.split(','))) for line in result.stdout.strip().splitlines()]
            available=[r for r in rows if r[0]==0 and r[1]<=10 and r[2]>=20000]
            save('queue_status.json',dict(next_arm=name,state='ready' if available else 'waiting_for_gpu0',gpu_rows=rows,checked_at=time.time()))
            if available:break
            time.sleep(90)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',
            LLM_MODEL_NAME='qwen3.8-flash',LLM_MODEL_NAMES='qwen3.8-flash',PYTHONUNBUFFERED='1')
        command=['bash','scripts/run_with_llm_provider.sh','dashscope',PY,'experiments/skills/run_group.py',
                 '--skills',skills,'--candidate-expansion',expansion,'--group','riskweaver_profile_off',
                 '--manifest',str(OUT/'manifest.json'),'--output-dir',str(OUT/name),
                 '--model','qwen3.8-flash','--models','qwen3.8-flash','--policy','idm','--seed','42','--steps','400']
        with (OUT/f'{name}.log').open('xb') as log:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(f'{name}_pid.json',dict(pid=p.pid,started_at=time.time(),command=command))
            code=p.wait()
        records.append(dict(arm=name,exit_code=code))
        save('launcher_result.partial.json',dict(groups=records))
        # 服务失败终止后续组，避免免费额度失效后制造一串无攻击结果。
        results=OUT/name/'results.json'
        if results.exists() and json.loads(results.read_text()).get('aggregate',{}).get('planner_service_failure_count',0):
            save('launcher_result.json',dict(groups=records,status='final_planner_service_failure'));return
    save('launcher_result.json',dict(groups=records,status='completed',finished_at=time.time()))


if __name__=='__main__':main()
