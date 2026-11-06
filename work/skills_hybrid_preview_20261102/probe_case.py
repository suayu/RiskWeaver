"""隔离帧间交互校验的生产装配复现；固定候选，禁止真实API和env.step。"""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import json
import sys

import numpy as np
from omegaconf import OmegaConf,open_dict

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'work/skills_temporal_candidates_20261029')]
from temporal_planner import TemporalCandidatePlanner
from temporal_intents import guarded_fractional_validator
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.hybrid_skills_planner import DynamicFirstStaticPlanner
from test_ego_profile import environment


def build_case():
    cfg=OmegaConf.load(ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
    with open_dict(cfg.sim.llm.skills):
        cfg.sim.llm.skills.source_fact_contract_enabled=False
        cfg.sim.llm.skills.candidate_expansion_enabled=True
    env=environment();env.cfg=cfg
    env.data_dict['agent'][-1][0,:5]=[14.,16.,10.,-32.,np.arctan2(-32,10)]
    def forbidden(*a,**k):raise AssertionError('API_and_env_step_forbidden')
    env.step=forbidden
    low=LLMAdversarialPlanner(model_name='zero-api-preview-fixture',model_names=['zero-api-preview-fixture'],
        provider='dashscope',attack_mode='joint',client=NS(chat=NS(completions=NS(create=forbidden))))
    low.last_trace={};low.use_obstacles=False
    dyn=TemporalCandidatePlanner(cfg.sim,4,extra_recipes=True,rolling_intent=True,
        explicit_contract=True,deduplicate=True,fractional_interaction=True,
        velocity_continuation=True,motion_evidence=True)
    anchors=np.array([[14.,16.],[24.,-16.],[34.,-48.],[44.,-80.]])
    estimates,reason=dyn.builder.screen(env,0,anchors,[env.scenario_dict['route']])
    if reason is not None:raise ValueError('controlled_candidate_fails_original_screen:'+reason)
    candidate=dict(candidate_id='cross_between_integer_seconds',target_id=0,strategy='lane_change',
        attack_family='crossing_conflict',duration=10,anchors=anchors.tolist(),screened=True,**estimates)
    # 只隔离同一个已过原动力学/背景筛查的候选；不把fixture目录当新生成效果。
    dyn.expansion=False
    dyn.builder.build=lambda *a,**k:([copy.deepcopy(candidate)],[])
    dyn.builder.last_audit=[]
    return cfg,env,low,dyn,candidate


def inspect_case():
    cfg,env,low,dyn,candidate=build_case()
    state=env.get_state_for_planning();original=low._validate_attack_plan
    before=dyn.build_context(env,state,low)
    events=[];low._validate_attack_plan=guarded_fractional_validator(original,events)
    try:inside=dyn.build_context(env,state,low)
    finally:low._validate_attack_plan=original
    branches=[]
    def dynamic_plan(*a,**k):branches.append('dynamic');return dict(attack=True)
    def static_plan(*a,**k):branches.append('static');return dict(attack=False)
    dyn.plan=dynamic_plan
    hybrid=DynamicFirstStaticPlanner(dyn,cfg.sim,4);hybrid.static.plan=static_plan
    hybrid.plan(low,env,state,[])
    return dict(original_preview_candidates=len(before['candidates']),
        under_plan_semantics_candidates=len(inside['candidates']),old_routing=branches,
        candidate=candidate,fractional_events=events,
        input=state,configuration_changes='source_fact_contract=False; controlled one-candidate builder',
        baseline_measurement_preflight=inside.get('measurement_preflight'),
        physical_API_requests=0,env_steps=0,D=None,collision=None,goal_complete=False)


if __name__=='__main__':print(json.dumps(inspect_case(),ensure_ascii=False,allow_nan=False))
