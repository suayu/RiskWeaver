"""用已完成74号源帧11做零API候选重放；只比较高层候选，不伪造闭环执行。"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
RUN=ROOT/'experiments/skills_rolling_contract_scene074_20261029_r1'
sys.path.insert(0,str(ROOT/'work/skills_rolling_intent_20261028'))
from temporal_audit import continuity
from temporal_intents import velocity_aligned_candidates,fractional_minimum,guarded_fractional_validator


def source_fixture():
    from omegaconf import OmegaConf
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from policies.ego_profile import CandidateBuilder
    trace=next((RUN/'contract_full').rglob('execution_trace.jsonl'))
    rows=[json.loads(s) for s in trace.read_text().splitlines()]
    state=next(r['state'] for r in rows if r.get('kind')=='llm_input' and r['step']==11)
    frame=next(r for r in rows if r.get('kind')=='state' and r['step']==11)
    accepted=next(r for r in rows if r.get('kind')=='llm_output' and r['step']==11)
    prior=accepted['planner_trace']['context']['rolling_intent']['previous']['previous']
    cfg=OmegaConf.load(RUN/'contract_full/resolved_config.yaml');route=np.asarray(state['route'])
    env=NS(ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
        agent_active=np.asarray(frame['active']),current_step=11,current_scene_id='9_4.pkl',
        scenario_dict={'route':route,'lanes':[route]},get_static_obstacles=lambda:[],cfg=cfg,dt=.1,
        pending_joint_trajectory=None,get_state_for_planning=lambda:state)
    planner=object.__new__(LLMAdversarialPlanner);planner.allow_trajectory=True;planner.use_obstacles=False
    planner.last_request_failed=False;planner.last_trace={}
    return env,state,planner,CandidateBuilder(cfg.sim.ego_profile),prior,accepted,trace


def replay():
    from policies.intent_validation import validate_world_intent
    env,state,planner,builder,prior,old,trace=source_fixture();source_hash=hashlib.sha256(trace.read_bytes()).hexdigest()
    proposals,fit=velocity_aligned_candidates(prior,env.data_dict['agent'][-1][prior['target_id']],11)
    result=[]
    for i,c in enumerate(proposals):
        c=dict(c,candidate_id=f'vc_{i}')
        estimate,screen=builder.screen(env,c['target_id'],np.asarray(c['anchors']),env.scenario_dict['lanes'])
        original=validate_world_intent(planner,state,c);events=[];original_method=planner._validate_attack_plan
        planner._validate_attack_plan=guarded_fractional_validator(original_method,events)
        try:refined=validate_world_intent(planner,state,c)
        finally:del planner._validate_attack_plan
        comparison=continuity(prior,dict(source_step=11,target_id=c['target_id'],anchors=c['anchors']),env.data_dict['agent'][-1][c['target_id']][:2])
        result.append(dict(candidate=c,screen_rejection=screen,original_semantic=original,fractional_semantic=refined,
            fractional_validation_events=events,fractional_active=fractional_minimum(c['anchors'],state['ego_state'],.1,1.),
            continuity=comparison,estimated_screen_metrics=estimate))
    old_continuations=[r for r in old['planner_trace']['context']['rolling_intent']['proposal_audit']
        if r['candidate'].get('recipe')=='remaining_world_intent_plus_fresh_tail_v1']
    assert hashlib.sha256(trace.read_bytes()).hexdigest()==source_hash
    return dict(version='074_velocity_continuation_zero_API_replay_v1',fit=fit,candidates=result,
        old_remaining_candidates=len(old_continuations),old_all_rejected_dynamics=all(r['screen_rejection']=='dynamics' for r in old_continuations),
        input_trace_sha256=source_hash,inputs_unchanged=True,
        lanes_scope='screen_current_attacker_has_no_lane_gate; exact_source_states_static_objects_and_margin_used',
        new_API_requests=0,new_execution_frames=0,new_diffusion_samples=0,
        closed_loop_effect_unverified=True,goal_complete=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=replay()
    with a.output.open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(old_dynamics_rejected=r['old_remaining_candidates'],candidates=[dict(id=x['candidate']['candidate_id'],
        screen=x['screen_rejection'],original_semantic=x['original_semantic']['accepted'],
        refined_semantic=x['fractional_semantic']['accepted'],active_gap=x['fractional_active']['min_center_gap_m'],
        active_time=x['fractional_active']['relative_time_seconds'],
        overlap_jump=x['continuity']['max_raw_overlap_deviation_m']) for x in r['candidates']],new_API_requests=0)))


if __name__=='__main__':main()
