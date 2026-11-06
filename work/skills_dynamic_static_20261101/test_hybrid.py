"""有车/无可用车/无道路空间及预算隔离；模型传输使用替身。"""
from pathlib import Path
import copy,json,sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(Path(__file__).resolve().parent)]
from test_ego_profile import environment
from test_bailian_profile_joint import obstacle_planner
from policies.skills_planner import SkillsAttackPlanner
from hybrid_planner import DynamicFirstStaticPlanner,create_planner
from static_catalog import build_static_combinations,validate_placements,free_area
from hybrid_rag import HybridAttackRAGRuntime


def setup(active=False,budget=4):
    env=environment();env.agent_active[:]=active
    old=env.get_state_for_planning
    env.get_state_for_planning=lambda:dict(old(),agents=old()['agents'] if env.agent_active.any() else [])
    sim=NS(ego_profile=NS(),llm=NS(skills=NS(enabled=True,max_repairs=1)))
    planner=obstacle_planner();planner.last_request_failed=False;planner.last_error=None
    dynamic=SkillsAttackPlanner(sim,budget);engine=DynamicFirstStaticPlanner(dynamic,sim,budget)
    calls=[]
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);calls.append(payload)
        ids=[c['candidate_id'] for c in payload['context']['candidates']]
        result={k:payload[k] for k in ['request_id','state_hash','catalog_hash']}
        result.update(ranking=ids,scene_analysis='Choose the published route obstruction or current vehicle intent.',
            candidate_reasons={cid:'Published interaction evidence.' for cid in ids},
            no_attack_reason='' if ids else 'No usable dynamic vehicle; choose static route obstruction.')
        if planner.use_obstacles:
            result['obstacle_candidate_id']=payload['context']['obstacle_candidates'][0]['obstacle_candidate_id']
        return '',result
    planner._request_attack_plan=request
    return env,planner,engine,calls


def test_empty_vehicle_scene_produces_multiple_types_and_whole_combinations():
    env,planner,engine,_=setup()
    catalog,audit=build_static_combinations(env,planner,engine.static.builder)
    assert catalog and any(len(c['placements'])==2 for c in catalog)
    assert any('construction_zone' in c['template_types'] for c in catalog)
    assert len({t for c in catalog for t in c['template_types']})>=4
    assert all(c['geometric_free_area_m2']>0 for c in catalog)
    assert any(not c['accepted'] for c in audit)


def test_dynamic_has_priority_and_does_not_send_static_fields():
    env,planner,engine,calls=setup(True)
    plan=engine.plan(planner,env,env.get_state_for_planning(),['challenge'])
    assert plan and plan['attack'] and plan['attack_kind']=='dynamic'
    assert 'obstacle_candidate_id' not in calls[0]
    assert 'obstacle_candidates' not in calls[0]['context']
    assert engine.static.calls==0 and engine.calls==1
    assert planner.use_obstacles is True


def test_no_vehicle_selects_complete_static_combination():
    env,planner,engine,calls=setup()
    plan=engine.plan(planner,env,env.get_state_for_planning(),['challenge'])
    assert plan and plan['static_attack'] and plan['attack_kind']=='static'
    assert plan['attack'] is False and len(plan['obstacle_plan'])==2
    assert not calls[0]['context']['candidates'] and engine.dynamic.calls==0
    assert engine.calls==1 and planner.last_trace['hybrid_routing']['branch']=='static'
    assert any(d['path']=='rw-static-fallback/SKILL.md' for d in calls[0]['knowledge']['documents'])


def test_visible_but_no_usable_dynamic_vehicle_still_gets_static_branch():
    env,planner,engine,calls=setup(True)
    env.data_dict['agent'][-1][0,:2]=[70,20]
    plan=engine.plan(planner,env,env.get_state_for_planning(),['challenge'])
    assert plan and plan['static_attack']
    assert planner.last_trace['hybrid_routing']['dynamic_candidate_count']==0


def test_background_collision_rejects_whole_obstacle_combination():
    env,planner,engine,_=setup(True)
    placement=dict(type='transverse_barrier',center=[25,0],yaw=0,count=2,spacing=1.2)
    result=validate_placements(env,planner,engine.static.builder,[placement])
    assert result['accepted'] is False and result['reason']=='background_CV_clearance'


def test_zero_geometric_area_is_not_accepted():
    env,planner,engine,_=setup()
    placement=dict(type='cone_line',center=[25,0],yaw=0,count=1,spacing=1)
    grid=dict(valid=True,points=np.empty((0,2)),resolution=.5,basis='empty_local_road')
    result=validate_placements(env,planner,engine.static.builder,[placement],grid)
    assert not result['accepted'] and result['reason']=='nonpositive_or_unknown_geometric_free_area'


def test_missing_forward_road_remains_explicit_unattackable_without_API():
    env,planner,engine,calls=setup()
    short=np.array([[0.,0.],[5.,0.]])
    env.scenario_dict=dict(route=short,lanes=np.array([short]))
    plan=engine.plan(planner,env,env.get_state_for_planning(),['challenge'])
    assert plan and not plan.get('obstacle_plan') and not plan['attack']
    assert plan['reason']=='no_feasible_candidates' and not calls


def test_service_failure_is_not_disguised_as_absent_vehicle_or_static_success():
    env,planner,engine,_=setup(True)
    from policies.llm_adversarial_planner import PlannerServiceFailure
    planner._request_attack_plan=lambda *a,**k:(_ for _ in ()).throw(PlannerServiceFailure('fixture service failure'))
    assert engine.plan(planner,env,env.get_state_for_planning(),[]) is None
    assert planner.last_request_failed and engine.static.calls==0
    assert planner.last_trace['hybrid_routing']['branch']=='dynamic'


def test_unknown_static_ID_is_rejected_without_rewriting_the_response():
    env,planner,engine,_=setup()
    original=planner._request_attack_plan
    def request(*a,**k):
        reason,response=original(*a,**k);response['obstacle_candidate_id']='invented';return reason,response
    planner._request_attack_plan=request
    assert engine.plan(planner,env,env.get_state_for_planning(),[]) is None
    assert not planner.last_request_failed and engine.calls==2
    assert planner.last_trace['requests'][0]['response']['obstacle_candidate_id']=='invented'


def test_two_branches_share_the_scene_budget():
    env,planner,engine,calls=setup(budget=2)
    assert engine.plan(planner,env,env.get_state_for_planning(),[])['static_attack']
    env.agent_active[:]=True
    assert engine.plan(planner,env,env.get_state_for_planning(),[])['attack']
    result=engine.plan(planner,env,env.get_state_for_planning(),[])
    assert result['reason']=='skills_api_budget_exhausted' and len(calls)==2


def test_disabled_factory_returns_original_planner_exactly():
    cfg=OmegaConf.create(dict(llm=dict(skills=dict(static_fallback_enabled=False))))
    result=object();calls=[]
    def factory(*a,**k):calls.append((a,k));return result
    assert create_planner(factory,cfg,4,extra_recipes=True) is result
    assert len(calls)==1


def test_static_RAG_intent_is_requested_but_not_fabricated_execution():
    runtime=HybridAttackRAGRuntime.__new__(HybridAttackRAGRuntime)
    runtime.pending=dict(actual_attack_execution_frames=0,actual_obstacle_exposure_frames=0)
    plan=dict(attack=False,obstacle_plan=[dict(type='cone_line')],reason='static choice')
    runtime.planned(plan,{})
    assert runtime.pending['requested_attack'] is True
    assert runtime.pending['attack_kind']=='static'
    assert runtime.pending['actual_attack_execution_frames']==0
    assert runtime.pending['actual_obstacle_exposure_frames']==0


def test_real_static_backend_objects_are_recorded_as_IDs():
    from policies.obstacles import ObstacleCatalog
    objects=ObstacleCatalog().materialize(dict(type='cone_line',center=[25,0],yaw=0,count=1,spacing=1),new_id=lambda name:'actual-created-1')
    runtime=HybridAttackRAGRuntime.__new__(HybridAttackRAGRuntime)
    runtime.pending=dict(accepted_intent=False,actual_attack_execution_frames=0,actual_obstacle_exposure_frames=0)
    runtime.mark_obstacles(objects)
    assert runtime.pending['obstacle_created_ids']==['actual-created-1']
    assert runtime.pending['accepted_intent'] is True
    assert runtime.pending['actual_obstacle_exposure_frames']==0


def test_final_factory_is_wrapped_after_existing_planner_assembly(monkeypatch):
    from experiments.skills import run_instrumented
    from experiments.comparison import run_e1_group as runner
    from hybrid_assembly import install
    env,planner,engine,_=setup()
    cfg=NS(sim=NS(llm=NS(skills=NS(static_fallback_enabled=True,max_repairs=1)),ego_profile=NS()))
    class FinalFactory:
        def __init__(self):
            self.cfg=cfg;self.skills_planner=engine.dynamic
            self._max_call_times=4;self.llm_planner=planner
    monkeypatch.setattr(runner,'AdversarialScenarioGenerator',FinalFactory)
    monkeypatch.setattr(runner,'group_overrides',lambda *a,**k:['base=value'])
    def actual_entry():
        instance=runner.AdversarialScenarioGenerator()
        return instance,runner.group_overrides()
    monkeypatch.setattr(run_instrumented,'main',actual_entry)
    with install():
        instance,overrides=run_instrumented.main()
    assert isinstance(instance.skills_planner,DynamicFirstStaticPlanner)
    assert instance.llm_planner.use_obstacles
    assert '++sim.llm.skills.static_fallback_enabled=true' in overrides
    assert 'sim.llm.attack_mode=joint' in overrides
    assert run_instrumented.main is actual_entry
    assert runner.AdversarialScenarioGenerator is FinalFactory


def test_public_generator_materializes_static_combination(monkeypatch):
    from test_generator_wiring_unit import fixture
    from policies.obstacle_wrapper import ObstacleWrapper,ScenarioDreamerObstacleBackend
    env,planner,engine,calls=setup()
    env.static_obstacle_elements={}
    wrapper=ObstacleWrapper(ScenarioDreamerObstacleBackend(env),planner.obstacle_catalog)
    env.apply_obstacle_plan=wrapper.create_plan;env.get_static_obstacles=wrapper.public_state
    generator=fixture();generator.llm_planner=planner;generator.skills_planner=engine
    generator.profile_enabled=False;generator.profile_pipeline=None;generator.difficulty_mode='off'
    generator.plan_and_inject(env)
    assert env.get_static_obstacles()
    assert generator._episode_obstacle_plan_count==1
    assert generator._episode_attack_plan_count==0
    assert env.attack_intent is None and len(calls)==1
