"""同目录/同技能/同严格校验的传输视图对照；先等实际单场终态，避免并发服务干扰。"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(ROOT/'work/skills_candidate_local_20261030')]
import run_development as development
from factored_view import install_factored_view
base=development.base

ARMS=['original_wire_view','factored_wire_view']


def run_trial(sample,mode,pool,cfg,out,planner_factory=None):
    if mode not in ARMS:raise ValueError('unknown_wire_view_arm')
    original=base.engine_for
    def engine_for(name,config,record):
        engine=development.engine_for('temporal_candidate_local',config,record)
        return install_factored_view(engine,name=='factored_wire_view')
    base.engine_for=engine_for
    try:
        kwargs={} if planner_factory is None else {'planner_factory':planner_factory}
        return base.run_trial(sample,mode,pool,cfg,out,**kwargs)
    finally:base.engine_for=original


def preceding_actual_task():
    run=ROOT/'experiments/skills_candidate_local_closed_loop_scene074_20261030_r2'
    terminal=run/'launcher_result.json';pid_file=run.with_suffix('.pid')
    if pid_file.exists():
        pid=int(pid_file.read_text());p=Path('/proc')/str(pid)/'cmdline'
        if p.exists() and 'queue_smoke.py' in p.read_bytes().decode():return dict(passed=False,reason='preceding_actual_task_live')
    if not terminal.exists():return dict(passed=False,reason='preceding_actual_task_not_terminal')
    status=json.loads(terminal.read_text())
    trace=run/'temporal_local/movies/scenario_074/execution_trace.jsonl'
    rows=list(map(json.loads,trace.read_text().splitlines())) if trace.exists() else []
    outputs=[r for r in rows if r.get('kind')=='llm_output' and (r.get('validated_plan') or {}).get('attack')]
    factory=any((r.get('planner_trace') or {}).get('common_identity_evidence') and
        (r.get('planner_trace') or {}).get('candidate_grounding_validation',{}).get('policy')=='candidate_local_strict' for r in outputs)
    return dict(passed=bool(status['status']=='completed' and status.get('source_verified') and factory),
        actual_highlevel_factory_verified=factory,prior_status=status['status'],scope='request_measurement_readiness_not_full_episode_or_threat_success')


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--request-view',choices=['legacy','paired'],default='legacy');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--free-protection-confirmed',action='store_true');a=p.parse_args()
    if a.request_view!='paired':raise ValueError('explicit_paired_view_required; legacy_planner_entry_is_unchanged')
    protocol=json.loads((a.protocol/'protocol.json').read_text());hashes=json.loads((a.protocol/'source_sha256.json').read_text())
    check=base.verify_sources(protocol,hashes);ready=preceding_actual_task()
    if a.prepare_only:print(json.dumps(dict(source_verified=check['verified'],predecessor=ready,new_API_calls=0)));return
    if not ready['passed'] or not check['verified'] or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not a.free_protection_confirmed:
        raise ValueError('view_experiment_not_ready_no_requests_sent')
    inputs=a.protocol/'inputs.json'
    if hashlib.sha256(inputs.read_bytes()).hexdigest()!=protocol['inputs_sha256']:raise ValueError('frozen_development_inputs_changed')
    samples=json.loads(inputs.read_text())
    if [s['scene'] for s in samples]!=[53,56,74,182] or protocol['arms']!=ARMS:raise ValueError('frozen_four_development_states_changed')
    if a.output is None:raise ValueError('explicit_new_output_required')
    a.output.mkdir(exist_ok=False)
    for name in ['protocol.json','inputs.json','resolved_config.yaml','source_sha256.json']:shutil.copyfile(a.protocol/name,a.output/name)
    shutil.copytree(a.protocol/'source',a.output/'source_snapshot');a.output.with_suffix('.pid').write_text(str(os.getpid()))
    base.save(a.output/'launch.json',dict(pid=os.getpid(),started_at=time.time(),planned_state_arms=8,maximum_physical_calls=32,actual_GPU_work=False,predecessor=ready))
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(a.protocol/'resolved_config.yaml');rows=[];checks=[];undo=base.install_control('service_only')
    try:
        for index,sample in enumerate(samples):
            for mode in (ARMS if index%2==0 else list(reversed(ARMS))):
                check=base.verify_sources(protocol,hashes);checks.append(check);base.save(a.output/'runtime_source_integrity.json',checks)
                if not check['verified']:raise ValueError('source_drift_during_view_pair')
                row=run_trial(sample,mode,protocol['frozen_model_pool'],cfg,a.output/f'scene{sample["scene"]:03d}_{mode}')
                rows.append(row);base.save(a.output/'results.partial.json',dict(planned_state_arms=8,results=rows))
                if row.get('runtime_failure') or row.get('planner_service_failure'):raise RuntimeError('final_planning_or_runner_failure')
        base.save(a.output/'results.json',dict(status='completed',planned_state_arms=8,results=rows,
            final_source_check=base.verify_sources(protocol,hashes),new_execution_frames=0,D=None,
            compressed_reading_and_LLM_prose_still_require_review=True,goal_complete=False))
    except Exception as error:
        base.save(a.output/'results.json',dict(status='failed',error_type=type(error).__name__,planned_state_arms=8,
            results=rows,unattempted_state_arms=8-len(rows),new_execution_frames=0,D=None,goal_complete=False));raise
    finally:undo()


if __name__=='__main__':main()
