"""同版本原留出十初态真实百炼；不改预定分母、不启动闭环、不使用开发回复。"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT/'work/skills_candidate_local_20261030'))
import run_development as development
base=development.base
from omegaconf import OmegaConf

ARMS=['legacy_off','temporal_candidate_local']
EXPECTED=[6,28,45,102,114,143,146,185,229,232]


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--free-protection-confirmed',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or not a.free_protection_confirmed:raise ValueError('confirmed_free_CPU_only')
    protocol=json.loads((a.protocol/'protocol.json').read_text());hashes=json.loads((a.protocol/'source_sha256.json').read_text())
    preparation=json.loads((a.protocol/'stage_result.json').read_text())
    if protocol['arms']!=ARMS or protocol['scenes']!=EXPECTED or preparation['status']!='completed' or not protocol['development_gate']['passed']:
        raise ValueError('holdout_readiness_failed')
    if not base.verify_sources(protocol,hashes)['verified']:raise ValueError('holdout_source_changed')
    inputs=a.protocol/'inputs.json'
    if hashlib.sha256(inputs.read_bytes()).hexdigest()!=protocol['inputs_sha256']:raise ValueError('holdout_inputs_changed')
    samples=json.loads(inputs.read_text())
    if len(samples)!=10 or [s['scene'] for s in samples]!=EXPECTED:raise ValueError('original_holdout_denominator_changed')
    a.output.mkdir(exist_ok=False)
    for name in ['protocol.json','inputs.json','resolved_config.yaml','source_sha256.json','validation10.json']:shutil.copyfile(a.protocol/name,a.output/name)
    shutil.copytree(a.protocol/'source',a.output/'source_snapshot');a.output.with_suffix('.pid').write_text(str(os.getpid()))
    base.save(a.output/'launch.json',dict(pid=os.getpid(),started_at=time.time(),planned_state_arms=20,arms=ARMS,
        physical_calls_per_state_arm=4,maximum_total_physical_calls=80,actual_GPU_work=False,new_execution_frames=0))
    cfg=OmegaConf.load(a.protocol/'resolved_config.yaml');rows=[];checks=[];restore=base.install_control('service_only')
    try:
        for index,sample in enumerate(samples):
            for mode in (ARMS if index%2==0 else list(reversed(ARMS))):
                check=base.verify_sources(protocol,hashes);checks.append(check);base.save(a.output/'runtime_source_integrity.json',checks)
                if not check['verified']:raise ValueError('source_changed_during_holdout')
                row=base.run_trial(sample,mode,protocol['frozen_model_pool'],cfg,a.output/f'scene{sample["scene"]:03d}_{mode}')
                rows.append(row);base.save(a.output/'results.partial.json',dict(planned_state_arms=20,results=rows))
                if row.get('runtime_failure') or row.get('planner_service_failure'):raise RuntimeError('final_planning_or_runner_failure')
        base.save(a.output/'results.json',dict(status='completed',planned_state_arms=20,results=rows,
            final_source_check=base.verify_sources(protocol,hashes),new_execution_frames=0,D=None,
            original_holdout_initial_states_only=True,majority_correct_planning_certified=False,goal_complete=False))
    except Exception as error:
        base.save(a.output/'results.json',dict(status='failed',error_type=type(error).__name__,planned_state_arms=20,results=rows,
            unattempted_state_arms=20-len(rows),new_execution_frames=0,D=None,goal_complete=False));raise
    finally:restore()


if __name__=='__main__':main()
