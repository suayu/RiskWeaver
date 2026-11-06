"""显式启动入口；未通过单场门禁时不创建实验目录或派发任何场景。"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'work/skills_modelpool_budget_dev_20261022'),
    str(ROOT/'work/skills_protocol_preparation_20261025'),str(ROOT/'work/skills_shared_formal_20261028'),
    str(ROOT/'work/skills_shared_resource_20261028')]
from validation_core import readiness,rewrite_command


def check_prerequisites(protocol,metadata):
    from audit_four_arm import smoke_prerequisite
    from audit_shared import audit_shared
    from launch_pool_pair import verify_sources
    dev=ROOT/metadata['development_experiment'];smoke=ROOT/metadata['smoke_experiment']
    manifest=json.loads((protocol/'validation10.json').read_text())
    development=audit_shared(dev)['development_gate']
    single=smoke_prerequisite(smoke,'contract_full',74)
    source=verify_sources(metadata,json.loads((protocol/'source_sha256.json').read_text()))
    result=readiness(development,single,source,manifest)
    return dict(result,development=development,smoke=single,source=source)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output-dir',type=Path);p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();metadata=json.loads((args.protocol/'protocol.json').read_text())
    checks=check_prerequisites(args.protocol,metadata)
    if args.prepare_only:
        print(json.dumps(checks,ensure_ascii=False));return checks
    if not checks['passed']:raise ValueError('heldout_prerequisites_not_passed; no_scenarios_dispatched')
    if args.output_dir is None:raise ValueError('explicit_new_output_directory_required')
    if args.output_dir.exists():raise FileExistsError('preserve_prior_validation')
    import launch_pool_pair as base
    import pool_acceptance
    from resource_policy import eligible_shared
    from audit_validation import audit_validation
    original=(base.idle_gpus,base.child_command,pool_acceptance.audit_pool,base.save);argv=sys.argv
    def idle(reserved):
        if reserved:return []
        if shutil.disk_usage(ROOT).free<5.5*2**30:raise RuntimeError('insufficient_disk_user_action_required')
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15)
        return [row['index'] for row in eligible_shared(rows,14.,3.,allowed=('0',))][:1]
    def command(prepared,manifest,output):return rewrite_command(prepared['command'],manifest,output,metadata['frozen_model_pool'])
    def save(path,value):
        if path.name=='launch.json':
            value=dict(value,planned_attempts=20,planned_scenes_per_arm=10,
                heldout_prerequisites=checks,shared_resource=dict(gpu='0',single_own_arm=True,cap_gib=14,reserve_gib=3),
                exposure_scope='not_used_in_this_skills_development_not_project_unseen')
        if path.name=='queue_status.json':value=dict(value,state='waiting_for_GPU0_17GiB_headroom')
        return original[3](path,value)
    base.idle_gpus=idle;base.child_command=command;pool_acceptance.audit_pool=audit_validation;base.save=save
    try:
        sys.argv=[sys.argv[0],'--protocol',str(args.protocol),'--stage','validation','--output-dir',str(args.output_dir),
            '--development-result',str(ROOT/metadata['development_experiment']),'--free-protection-confirmed']
        return base.main()
    finally:
        sys.argv=argv;base.idle_gpus,base.child_command,pool_acceptance.audit_pool,base.save=original


if __name__=='__main__':main()
