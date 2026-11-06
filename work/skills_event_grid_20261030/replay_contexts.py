"""原开发10场冻结初态的全链候选重放；无API、无扩散采样、无实际执行。"""
from pathlib import Path
import argparse,copy,hashlib,json,os,random,sys,time
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_temporal_candidates_20261029')]
from event_planner import FutureEventPlanner
from temporal_planner import TemporalCandidatePlanner


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if a.output.exists():raise FileExistsError('preserve_previous_context_replay')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only')
    import torch
    from omegaconf import OmegaConf
    for key in ['PROJECT_ROOT','SCRATCH_ROOT']:os.environ.setdefault(key,str(ROOT))
    os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'));torch.set_num_threads(2)
    from simulator import Simulator
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from policies.intent_validation import canonical_hash
    run=ROOT/'experiments/skills_temporal_development10_paired_20261029_r1'
    manifest=json.loads((run/'manifest.json').read_text())
    expected=[53,56,74,80,103,140,181,182,207,220]
    if manifest['original_scenario_indices']!=expected:raise ValueError('original_development_scene_denominator_changed')
    cfg=OmegaConf.load(run/'skills_full_shared_recovery/resolved_config.yaml')
    cfg.sim.traffic_model.device='cpu';cfg.sim.traffic_model.multi_gpu_devices=[]
    cfg.sim.visualize=False;cfg.sim.save_carla_data=False
    cfg.sim.scenario_data_output_path=str(a.output/'diagnostic_initial_data');cfg.sim.movie_path=str(a.output/'unused_movies')
    dependencies=json.loads((run/'source_sha256.json').read_text())
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in dependencies}
    for f in HERE.glob('*.py'):hashes[f.relative_to(ROOT).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
    for name,digest in dependencies.items():
        if hashes[name]!=digest:raise ValueError('original_frozen_source_changed:'+name)
    a.output.mkdir();(a.output/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    (a.output/'launch.json').write_text(json.dumps(dict(planned_scenes=expected,arms=['temporal_original','full_grid'],
        planned_state_arms=20,scope='existing_development_initial_states_not_generalization_or_LLM_performance',
        API_calls=0,diffusion_samples=0,execution_frames=0,started_at=time.time()),indent=2))
    def forbidden(*args,**kwargs):raise AssertionError('API_forbidden_in_offline_candidate_replay')
    LLMAdversarialPlanner._request_attack_plan=forbidden
    env=Simulator(cfg);env.test_files=manifest['scenario_files'];records=[]
    flags=dict(extra_recipes=True,rolling_intent=True,explicit_contract=True,deduplicate=True,fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
    for index,scene in enumerate(expected):
        effective=(42+1000003*scene)%(2**31-1);random.seed(effective);np.random.seed(effective);torch.manual_seed(effective)
        env.reset(index);state=env.get_state_for_planning()
        trace=run/'legacy_shared_recovery/movies'/f'scenario_{scene:03d}'/'execution_trace.jsonl'
        rows=[json.loads(s) for s in trace.read_text().splitlines()]
        frame=next(r for r in rows if r.get('kind')=='state' and r['step']==0)
        if not np.array_equal(env.ego_state,np.asarray(frame['ego'])):raise ValueError('frozen_ego_initial_state_changed')
        if not np.allclose(env.data_dict['agent'][-1][env.agent_active],np.asarray(frame['agents'])[np.asarray(frame['active'])],atol=1e-6):raise ValueError('frozen_background_initial_state_changed')
        for mode in ['temporal_original','full_grid']:
            planner=object.__new__(LLMAdversarialPlanner);planner.allow_trajectory=True;planner.use_obstacles=False
            planner.last_request_failed=False;planner.last_trace={}
            engine=TemporalCandidatePlanner(cfg.sim,4,**flags) if mode=='temporal_original' else FutureEventPlanner(cfg.sim,4,event_grid=True,**flags)
            context=engine.build_context(env,state,planner)
            candidates=context['candidates'];events=[c for c in candidates if c.get('recipe')=='full_grid_source_CV_sparse_intercept_v1']
            visible_state={k:state[k] for k in ('ego_state','agents','route','current_step')}
            result=dict(scene=scene,mode=mode,state_hash=canonical_hash(visible_state),visible_candidate_count=len(candidates),
                visible_candidate_ids=[c['candidate_id'] for c in candidates],visible_event_ids=[c['candidate_id'] for c in events],
                event_grid_screened_pass_count=sum(bool(r.get('semantic_validation',{}).get('accepted')) for r in getattr(engine,'event_grid_audit',[])),
                measurement_preflight=context.get('measurement_preflight'),route_tail_preflight=context.get('route_tail_preflight'),
                context=context,event_grid_audit=getattr(engine,'event_grid_audit',[]),API_calls=0)
            path=a.output/f'scene{scene:03d}_{mode}.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
            records.append({k:v for k,v in result.items() if k not in ('context','event_grid_audit')})
            (a.output/'results.partial.json').write_text(json.dumps(dict(records=records),ensure_ascii=False,indent=2))
    intact=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    result=dict(records=records,source_unchanged=intact,planned_scenes=10,planned_state_arms=20,API_calls=0,diffusion_samples=0,execution_frames=0,
        scope='candidate_availability_only_not_correct_LLM_planning_or_actual_threat',goal_complete=False)
    (a.output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (a.output/'exit_status.json').write_text(json.dumps(dict(exit_code=0 if intact else 1)))
    print(json.dumps(dict(records=len(records),source_unchanged=intact,counts=[dict(scene=r['scene'],mode=r['mode'],count=r['visible_candidate_count'],new=len(r['visible_event_ids'])) for r in records])))


if __name__=='__main__':main()
