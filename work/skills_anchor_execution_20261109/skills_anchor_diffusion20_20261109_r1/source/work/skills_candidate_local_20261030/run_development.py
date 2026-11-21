"""三臂真实高层对照：原自由规划/全局严格/逐候选严格；不运行扩散或仿真。"""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,os,shutil,sys,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
spec=importlib.util.spec_from_file_location('original_highlevel_runner',ROOT/'work/skills_highlevel_comparison_20261030/frozen_planning.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from omegaconf import OmegaConf

ARMS=['legacy_off','temporal_global','temporal_candidate_local']
original_engine=base.engine_for


def engine_for(mode,cfg,sample):
    """参数只改变grounding失败隔离；两个skills臂使用相同目录、知识和原验证器。"""
    if mode not in ARMS:raise ValueError('unknown_candidate_local_comparison_arm')
    if mode=='legacy_off':return original_engine(mode,cfg,sample)
    local=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
    local.sim.llm.skills.grounding_validation_mode='candidate_local' if mode=='temporal_candidate_local' else 'global'
    return original_engine('temporal_skills_common',local,sample)


base.engine_for=engine_for


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--free-protection-confirmed',action='store_true');a=p.parse_args()
    if not a.free_protection_confirmed or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('confirmed_free_CPU_only_required')
    protocol=json.loads((a.protocol/'protocol.json').read_text());hashes=json.loads((a.protocol/'source_sha256.json').read_text())
    if protocol['arms']!=ARMS or not base.verify_sources(protocol,hashes)['verified']:raise ValueError('protocol_or_source_changed')
    inputs=a.protocol/'inputs.json'
    if hashlib.sha256(inputs.read_bytes()).hexdigest()!=protocol['inputs_sha256']:raise ValueError('inputs_changed')
    samples=json.loads(inputs.read_text());planned=len(samples)*len(ARMS)
    if len(samples)!=10 or [s['scene'] for s in samples]!=protocol['scenes'] or planned!=30:raise ValueError('ten_scene_three_arm_denominator_changed')
    if not (ROOT/protocol['free_protection_receipt']).is_file():raise ValueError('free_protection_receipt_missing')
    cfg=OmegaConf.load(a.protocol/'resolved_config.yaml');a.output.mkdir(exist_ok=False)
    for name in ['protocol.json','inputs.json','resolved_config.yaml','source_sha256.json']:shutil.copyfile(a.protocol/name,a.output/name)
    shutil.copytree(a.protocol/'source',a.output/'source_snapshot');a.output.with_suffix('.pid').write_text(str(os.getpid()))
    base.save(a.output/'launch.json',dict(pid=os.getpid(),started_at=time.time(),planned_state_arms=planned,arms=ARMS,
        physical_requests_per_state_arm=4,maximum_total_physical_requests=planned*4,new_execution_frames=0,actual_GPU_work=False))
    rows=[];checks=[];restore=base.install_control('service_only')
    try:
        for index,sample in enumerate(samples):
            # 循环轮换先后顺序；模型池相同但真实回退、token和时延仍须单独披露。
            order=ARMS[index%3:]+ARMS[:index%3]
            for mode in order:
                check=base.verify_sources(protocol,hashes);checks.append(check);base.save(a.output/'runtime_source_integrity.json',checks)
                if not check['verified']:raise ValueError('source_changed_during_development')
                result=base.run_trial(sample,mode,protocol['frozen_model_pool'],cfg,a.output/f'scene{sample["scene"]:03d}_{mode}')
                result['grounding_policy']='candidate_local' if mode=='temporal_candidate_local' else 'global' if mode=='temporal_global' else 'original_free_planner'
                base.save(a.output/f'scene{sample["scene"]:03d}_{mode}/result.json',result)
                rows.append(result);base.save(a.output/'results.partial.json',dict(planned_state_arms=planned,results=rows))
                if result.get('runtime_failure') or result.get('planner_service_failure'):raise RuntimeError('final_planner_or_runner_failure')
        base.save(a.output/'results.json',dict(status='completed',planned_state_arms=planned,results=rows,
            final_source_check=base.verify_sources(protocol,hashes),new_execution_frames=0,D=None,
            scope='development_initial_states_only_not_full_scene_or_correctness_certification',goal_complete=False))
    except Exception as error:
        base.save(a.output/'results.json',dict(status='failed',error_type=type(error).__name__,planned_state_arms=planned,results=rows,
            unattempted_state_arms=planned-len(rows),new_execution_frames=0,D=None,goal_complete=False));raise
    finally:restore()


if __name__=='__main__':main()
