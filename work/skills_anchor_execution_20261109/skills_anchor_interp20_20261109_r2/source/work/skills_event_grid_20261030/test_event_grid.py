"""完整网格、原拒绝不绕过及默认关闭/真实提示链回归；零真实API。"""
from types import SimpleNamespace as NS
import copy,json,sys
import numpy as np
import pytest
from event_grid import TIMES,screened_event_candidates
from event_planner import FutureEventPlanner,ROOT


def simple_env():
    states=np.array([[-2.,2.,0.,-1.,-np.pi/2,4.5,2.,1.]])
    return NS(ego_state=np.array([0.,0.,0.,1.,np.pi/2,4.5,2.,1.]),data_dict={'agent':[states]},
        scenario_dict={'lanes':[np.array([[0.,0.],[0.,10.]])]},agent_active=np.array([True]),current_step=0,dt=.1)


def test_all30_times_logged_no_more_than_two_published_per_target():
    env=simple_env();calls=[]
    builder=NS(screen=lambda *args:(dict(estimated_risk=.5),None))
    def semantic(state,c):calls.append(c);return dict(accepted=True,metrics={})
    values,audit=screened_event_candidates(env,{'agents':[{'id':0}]},builder,semantic)
    assert len(TIMES)==30 and len(audit)==30 and [r['requested_event_seconds'] for r in audit]==list(TIMES)
    assert 1<=len(values)<=2 and len(calls)>0
    for c in values:
        assert np.array_equal(c['anchors'][0],env.data_dict['agent'][-1][0,:2])
        assert c['interpolation_event_residual_m']<1e-9 and c['duration']==10
        assert c['event_within_current_lease']==(c['event_time_seconds']<=1.)


@pytest.mark.parametrize('failure',['screen','semantic'])
def test_original_screen_or_semantic_rejection_never_published(failure):
    builder=NS(screen=lambda *args:(None,'background_margin') if failure=='screen' else ({},None))
    values,audit=screened_event_candidates(simple_env(),{'agents':[{'id':0}]},builder,lambda *args:dict(accepted=False,reason='original_rule'))
    assert not values and len(audit)==30 and not any(r['published'] for r in audit)
    assert any(r.get('screen_rejection')=='background_margin' if failure=='screen' else r.get('semantic_validation',{}).get('reason')=='original_rule' for r in audit)


def source_fixture():
    sys.path.insert(0,str(ROOT/'work/skills_temporal_candidates_20261029'))
    from replay_074 import source_fixture as fixture
    return fixture()


def flags():return dict(extra_recipes=True,rolling_intent=True,explicit_contract=True,deduplicate=True,fractional_interaction=True,velocity_continuation=True,motion_evidence=True)


def test_default_off_is_exact_old_context_for_real_recorded_source():
    from temporal_planner import TemporalCandidatePlanner
    env,state,planner,*_=source_fixture()
    old=TemporalCandidatePlanner(env.cfg.sim,4,**flags());new=FutureEventPlanner(env.cfg.sim,4,**flags())
    assert new.build_context(env,state,planner)==old.build_context(env,state,planner)
    assert not new.event_grid_enabled


def test_boolean_and_required_temporal_path_enforced():
    env,*_=source_fixture()
    with pytest.raises(ValueError,match='boolean'):FutureEventPlanner(env.cfg.sim,4,event_grid=1,**flags())
    with pytest.raises(ValueError,match='temporal_expansion'):FutureEventPlanner(env.cfg.sim,4,event_grid=True)


def test_real_parent_prompt_pipeline_keeps_event_facts_and_same_contract(monkeypatch):
    env,state,planner,*_=source_fixture()
    engine=FutureEventPlanner(env.cfg.sim,4,event_grid=True,**flags());seen=[]
    sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
    from test_fact_runtime import answer
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);seen.append(payload)
        assert 'future_event_evidence' in payload['context'] and 'NOT a measured future' in prompt
        return '',answer(payload)
    planner._request_attack_plan=request;original=planner._validate_attack_plan
    result=engine.plan(planner,env,state,['high-level analysis only'])
    assert seen and result is not None and result['attack']
    assert planner._validate_attack_plan is original and planner._request_attack_plan is request
    assert planner.last_trace['future_event_protocol']['source_future_only']


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_entry_assembles_optional_factory_and_restores_it(tmp_path,monkeypatch,enabled):
    import run_event_controlled,run_temporal_controlled,temporal_planner
    env,*_=source_fixture();original=temporal_planner.TemporalCandidatePlanner;seen=[]
    def delegated():
        engine=temporal_planner.TemporalCandidatePlanner(env.cfg.sim,4,**flags())
        seen.append(engine)
        assert ('--future-event-grid' not in sys.argv) and '--prepare-only' in sys.argv
    monkeypatch.setattr(run_temporal_controlled,'main',delegated)
    argv=['entry','--output-dir',str(tmp_path/'new'),'--prepare-only']
    if enabled:argv+=['--future-event-grid','on','--skills','on','--rolling-intent','on','--fractional-interaction','on','--active-window-recipes','on']
    monkeypatch.setattr(sys,'argv',argv);run_event_controlled.main()
    assert bool(getattr(seen[0],'event_grid_enabled',False))==enabled
    assert temporal_planner.TemporalCandidatePlanner is original and sys.argv is argv
