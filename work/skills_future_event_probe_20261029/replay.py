"""原74源帧0的30时刻全网格复核；不安装到规划器、不改变当前311源码。"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
RUN=ROOT/'experiments/skills_temporal_candidates_scene074_20261029_r1'
sys.path[:0]=[str(ROOT/'work/skills_temporal_candidates_20261029')]
from sparse_event import solve_sparse_event


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve_prior_probe')
    from omegaconf import OmegaConf
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from policies.ego_profile import CandidateBuilder
    from policies.intent_validation import validate_world_intent
    trace=next((RUN/'temporal_full').rglob('execution_trace.jsonl'))
    frozen=json.loads((RUN/'source_sha256.json').read_text());paths=['policies/llm_adversarial_planner.py','policies/ego_profile.py','policies/intent_validation.py']
    # 模型源码可有额度注释；本次要求原始字节一致，不能在未重审时忽略漂移。
    for p in paths:
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=frozen[p]:raise ValueError('probe_dependency_changed:'+p)
    rows=[json.loads(s) for s in trace.read_text().splitlines()]
    state=next(r['state'] for r in rows if r.get('kind')=='llm_input' and r['step']==0)
    frame=next(r for r in rows if r.get('kind')=='state' and r['step']==0)
    cfg=OmegaConf.load(RUN/'temporal_full/resolved_config.yaml');route=np.asarray(state['route'])
    env=NS(ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},agent_active=np.asarray(frame['active']),
        current_step=0,current_scene_id='9_4.pkl',scenario_dict={'route':route,'lanes':[route]},
        get_static_obstacles=lambda:[],cfg=cfg,dt=.1,pending_joint_trajectory=None,get_state_for_planning=lambda:state)
    planner=object.__new__(LLMAdversarialPlanner);planner.allow_trajectory=True;planner.use_obstacles=False
    planner.last_request_failed=False;planner.last_trace={};builder=CandidateBuilder(cfg.sim.ego_profile)
    records=[];target=0
    for tick in range(1,31):
        time=tick/10;record=solve_sparse_event(frame['ego'],frame['agents'][target],time)
        record.update(original_recipe_time=time in [.5,1.,3.],active_lease_seconds=1.,event_outside_single_lease=time>1.)
        if record['feasible']:
            c=dict(target_id=target,strategy='lane_change',anchors=record['anchors'],duration=10,candidate_id=f'offline_event_{tick}')
            estimate,reason=builder.screen(env,target,np.asarray(c['anchors']),env.scenario_dict['lanes'])
            semantic=validate_world_intent(planner,state,c)
            record.update(original_screen_rejection=reason,original_semantic=semantic,screen_estimates=estimate,
                candidate=c,all_original_prechecks_passed=reason is None and semantic['accepted'])
        records.append(record)
    source_files=[trace,RUN/'temporal_full/resolved_config.yaml']+[ROOT/p for p in paths]
    result=dict(scope='single_development_source_state_exploratory_grid_not_new_attack_or_safety_certificate',
        source_step=0,target_id=0,original_recipe_event_seconds=[.5,1.,3.],all30_requested_event_times_preserved=True,
        records=records,input_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        source_only_ego_forecast=True,no_actual_future_states_used_to_construct_candidates=True,
        unchanged_sparse_acceleration_cap_mps2=3.8,original_background_and_semantic_checks_used=True,
        road_scope='attacker_road_gate_exempt; lanes_argument_unused_by_existing_screen',
        not_integrated_into_current_experiment=True,new_API_requests=0,new_diffusion_samples=0,new_execution_frames=0,goal_complete=False)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(output=str(args.output),event_times_passing_all_original_prechecks=[r['requested_event_seconds'] for r in records if r.get('all_original_prechecks_passed')],
        feasible_sparse_coefficient_times=[r['requested_event_seconds'] for r in records if r['feasible']],new_API_requests=0,new_execution_frames=0)))


if __name__=='__main__':main()
