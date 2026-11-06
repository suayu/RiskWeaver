"""新时序版专用阶段入口；不修改旧留出框架，未验收时零派发。"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT/'work/skills_modelpool_budget_dev_20261022'),
    str(ROOT/'work/skills_protocol_preparation_20261025'),str(ROOT/'work/skills_shared_formal_20261028'),
    str(ROOT/'work/skills_shared_resource_20261028'),str(ROOT/'work/skills_heldout_validation_20261029')]
from stage_core import readiness,same_temporal_runtime,validate_manifest
from validation_core import rewrite_command


def load(path):return json.loads(path.read_text()) if path.exists() else {}


def check_prerequisites(protocol,metadata):
    from launch_pool_pair import verify_sources
    from audit_four_arm import smoke_prerequisite
    from audit_stage import audit_stage
    stage=metadata['stage'];smoke=ROOT/metadata['smoke_experiment']
    hashes=load(protocol/'source_sha256.json');source=verify_sources(metadata,hashes)
    single=smoke_prerequisite(smoke,'temporal_full',74)
    development=None
    if stage=='validation':
        directory=ROOT/metadata['current_temporal_development_experiment']
        terminal=load(directory/'launcher_result.json');manifest=load(directory/'manifest.json')
        manifest_ok=False
        if manifest:
            try:validate_manifest(manifest,'development');manifest_ok=True
            except ValueError:pass
        audited=audit_stage(directory) if terminal.get('status')=='completed' and manifest_ok else {}
        development=dict(terminal_completed=terminal.get('status')=='completed',
            original_development_manifest_verified=manifest_ok,
            current_runtime_verified=same_temporal_runtime(load(directory/'source_sha256.json'),hashes),
            gate=audited.get('development_gate',{}))
    result=readiness(stage,single,source,load(protocol/f'{stage}10.json'),
        same_temporal_runtime(load(smoke/'source_sha256.json'),hashes),development)
    return dict(result,smoke=single,source=source,development=development)


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output-dir',type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    metadata=load(a.protocol/'protocol.json');checks=check_prerequisites(a.protocol,metadata)
    if a.prepare_only:print(json.dumps(checks,ensure_ascii=False));return checks
    if not checks['passed']:raise ValueError('temporal_stage_not_ready_no_scenarios_dispatched')
    if a.output_dir is None or not a.output_dir.is_absolute() or ROOT not in a.output_dir.parents:
        raise ValueError('explicit_new_workspace_output_required')
    if a.output_dir.exists():raise FileExistsError('preserve_prior_stage')
    import launch_pool_pair as base
    import pool_acceptance
    from resource_policy import eligible_shared
    from audit_stage import audit_stage
    prior=(base.idle_gpus,base.child_command,pool_acceptance.audit_pool,base.save);argv=sys.argv
    def idle(reserved):
        if reserved:return []
        if shutil.disk_usage(ROOT).free<5.5*2**30:raise RuntimeError('insufficient_disk_user_action_required')
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15)
        return [r['index'] for r in eligible_shared(rows,14.,3.,allowed=('0',))][:1]
    def command(prepared,manifest,output):return rewrite_command(prepared['command'],manifest,output,metadata['frozen_model_pool'])
    def save(path,value):
        if path.name=='launch.json':value=dict(value,temporal_prerequisites=checks,
            planned_scenes_per_arm=10,planned_attempts=20,shared_gpu='0',single_own_arm=True,
            exposure_scope='validation_not_used_this_skills_development_not_project_unseen' if metadata['stage']=='validation' else 'development_reused_for_tuning')
        if path.name=='queue_status.json':value=dict(value,state='waiting_for_GPU0_17GiB_headroom')
        return prior[3](path,value)
    base.idle_gpus=idle;base.child_command=command;pool_acceptance.audit_pool=audit_stage;base.save=save
    try:
        sys.argv=[sys.argv[0],'--protocol',str(a.protocol),'--stage',metadata['stage'],
            '--output-dir',str(a.output_dir),'--free-protection-confirmed']
        if metadata['stage']=='validation':sys.argv+=['--development-result',str(ROOT/metadata['current_temporal_development_experiment'])]
        return base.main()
    finally:sys.argv=argv;base.idle_gpus,base.child_command,pool_acceptance.audit_pool,base.save=prior


if __name__=='__main__':main()
