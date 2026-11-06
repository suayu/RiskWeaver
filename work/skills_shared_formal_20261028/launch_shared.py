"""探测通过后才允许单GPU共享派发；不更改原空闲队列或已运行进程。"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT/'work/skills_four_arm_launch_20261026'),str(ROOT/'work/skills_shared_resource_20261028')]
import launch as previous
from resource_policy import eligible_shared
from audit_shared import audit_shared


def shared_idle(reserved):
    # 原循环可容纳两个线程，但准入最多放行一个自己的GPU作业。
    if reserved:return []
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    return [row['index'] for row in eligible_shared(rows,14.,3.,allowed=('0',))][:1]


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--protocol',type=Path,required=True)
    args,_=parser.parse_known_args();protocol=json.loads((args.protocol/'protocol.json').read_text())
    probe=ROOT/protocol['shared_resource']['probe_experiment']
    final=json.loads((probe/'launcher_result.json').read_text())
    acceptance=json.loads((probe/'resource_acceptance.json').read_text())
    if final.get('status')!='resource_probe_passed' or not acceptance.get('passed'):
        raise ValueError('shared_resource_probe_not_passed')
    pid=json.loads((probe/'probe_pid.json').read_text())['pid'];p=Path('/proc')/str(pid)/'cmdline'
    if p.exists() and str(probe).encode() in p.read_bytes():raise ValueError('resource_probe_still_live')
    launcher=previous.previous.launcher
    original_idle=launcher.idle_gpus;original_audit=previous.previous.audit_four_arm;original_save=launcher.save
    def save(path,value):
        if path.name=='launch.json':
            value=dict(value,shared_resource=protocol['shared_resource'],resource_admission_concurrency=1,
                method_runtime_budget_unchanged=True,probe_outputs_not_merged=True)
        if path.name=='queue_status.json':
            # 共享队列仍检查显存准入，不能沿用旧队列“完全空闲”的状态描述。
            value=dict(value,state='waiting_for_shared_memory_headroom',gpu='0',required_free_gib=17.,max_active_arms=1)
        return original_save(path,value)
    launcher.idle_gpus=shared_idle;previous.previous.audit_four_arm=audit_shared;launcher.save=save
    try:return previous.main()
    finally:
        launcher.idle_gpus=original_idle;previous.previous.audit_four_arm=original_audit;launcher.save=original_save


if __name__=='__main__':main()
