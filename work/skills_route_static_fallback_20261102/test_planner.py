"""公共静态请求与最终复查使用同一后备规则；请求替身不调用真实SDK。"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(Path(__file__).resolve().parent)]
from planner import make_static
from policies.hybrid_skills_planner import StaticCombinationPlanner
from policies.intent_validation import canonical_hash


def test_real_request_and_final_validation_share_route_rule():
    spec=importlib.util.spec_from_file_location('route_static_plan_fixture',ROOT/'work/hybrid_public_integration_20261101/test_public_hybrid.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env,low,engine,calls=module.setup(False)
    env.scenario_dict['route']=np.array([[0.,0.],[10.,0.],[10.,100.]])
    env.scenario_dict['lanes']=np.array([env.scenario_dict['route']])
    cfg=NS(ego_profile=NS(),llm=NS(skills=NS(enabled=True,max_repairs=1,source_fact_contract_enabled=True)))
    engine.static=make_static(cfg,enabled=True)
    original=low._validate_obstacle_plan
    plan=engine.plan(low,env,env.get_state_for_planning(),['动态优先，无动态候选时选择静态组合'])
    assert plan['static_attack'] and engine.calls==len(calls)==1
    assert calls[0]['context']['static_clearance_policy']['route_footprint_entry_minimum_m']==15
    assert calls[0]['catalog_hash']==canonical_hash(calls[0]['context'])
    assert any(d['path']=='prototype/rw-route-static-clearance/SKILL.md' for d in calls[0]['knowledge']['documents'])
    assert any(e['accepted'] for e in low.last_trace['route_static_clearance']['events'])
    assert low._validate_obstacle_plan==original
    json.dumps(low.last_trace,allow_nan=False)


def test_disabled_factory_returns_exact_public_class():
    cfg=NS(ego_profile=NS(),llm=NS(skills=NS(enabled=True,max_repairs=1)))
    assert type(make_static(cfg,enabled=False)) is StaticCombinationPlanner
