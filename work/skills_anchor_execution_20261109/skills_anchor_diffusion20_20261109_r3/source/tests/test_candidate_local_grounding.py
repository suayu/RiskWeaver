"""真实时序提示/原验证/编译链：尾项错误隔离，身份和被选候选仍严格。"""
from pathlib import Path
import copy,json,sys
import pytest
from omegaconf import OmegaConf
from policies.candidate_grounding import CandidateGroundingFailure,validate_candidate_local

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/skills_temporal_candidates_20261029'))
from temporal_planner import TemporalCandidatePlanner
from replay_074 import source_fixture


def make_case(mode=None,mutation='tail_missing_after'):
    env,state,planner,*_=source_fixture()
    cfg=OmegaConf.create(OmegaConf.to_container(env.cfg,resolve=True))
    if mode is not None:cfg.sim.llm.skills.grounding_validation_mode=mode
    engine=TemporalCandidatePlanner(cfg.sim,4,extra_recipes=True,rolling_intent=True,
        explicit_contract=True,deduplicate=True,fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
    sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
    from test_fact_runtime import answer
    outputs=[]
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);response=answer(payload)
        # SDK替身原助手只返回一个候选；这里显式构造两个合法项，再注入被测错误。
        for candidate in payload['context']['candidates'][1:2]:
            single=copy.deepcopy(payload);single['context']['candidates']=[candidate]
            additional=answer(single)
            response['ranking']+=additional['ranking']
            response['candidate_reasons'].update(additional['candidate_reasons'])
            response['grounding'].update(additional['grounding'])
        ranked=response['ranking'];assert len(ranked)>1
        if mutation=='tail_missing_after':response['grounding'][ranked[-1]]['interaction_basis']='later_forecast_only'
        elif mutation=='first_wrong_role':response['grounding'][ranked[0]]['target_role']='ego'
        elif mutation=='all_wrong_role':
            for row in response['grounding'].values():row['target_role']='ego'
        elif mutation=='stale_identity':response['state_hash']='wrong'
        elif mutation=='unknown_id':
            old=ranked[-1];ranked[-1]='invented'
            response['grounding']['invented']=response['grounding'].pop(old)
            response['candidate_reasons']['invented']=response['candidate_reasons'].pop(old)
        elif mutation=='first_bad_time':response['grounding'][ranked[0]]['active_horizon_seconds']=3.
        elif mutation=='first_wrong_contrast':response['grounding'][ranked[0]]['contrast_direction']='invented'
        outputs.append(copy.deepcopy(response));return '',response
    planner._request_attack_plan=request
    return engine,planner,env,state,outputs


def test_default_global_keeps_original_failure_and_repair_budget():
    engine,planner,env,state,outputs=make_case()
    assert engine.grounding_validation_mode=='global'
    assert engine.plan(planner,env,state,['probe']) is None
    assert engine.calls==2 and planner.last_error=='priority_missing_interaction_evidence'
    assert 'candidate_grounding_validation' not in planner.last_trace


def test_invalid_tail_cannot_veto_valid_first_and_raw_response_not_rewritten():
    engine,planner,env,state,outputs=make_case('candidate_local')
    plan=engine.plan(planner,env,state,['probe']);assert plan and plan['attack'] and engine.calls==1
    assert plan['candidate_id']==outputs[0]['ranking'][0]
    decision=planner.last_trace['candidate_grounding_validation']
    bad=decision['candidates'][-1]
    assert not bad['accepted'] and bad['reason']=='priority_missing_interaction_evidence'
    assert planner.last_trace['requests'][0]['response']==outputs[0]
    assert planner.last_trace['response_summary']==outputs[0]


@pytest.mark.parametrize('mutation,reason',[('first_wrong_role','priority_actor_role_mismatch'),
    ('first_bad_time','priority_active_horizon_mismatch'),('first_wrong_contrast','incremental_direction_contradicts_fact')])
def test_invalid_first_never_compiled_and_original_valid_second_is_selected(mutation,reason):
    engine,planner,env,state,outputs=make_case('candidate_local',mutation)
    plan=engine.plan(planner,env,state,['probe'])
    assert plan and plan['candidate_id']==outputs[0]['ranking'][1] and engine.calls==1
    rejected=planner.last_trace['compile_attempts'][0]
    assert rejected['accepted'] is False and rejected['reason']==reason and rejected['stage']=='candidate_grounding'


@pytest.mark.parametrize('mutation,reason',[('stale_identity','skills_identity_mismatch'),('unknown_id','skills_ranking_not_catalog')])
def test_global_identity_and_unknown_ids_cannot_be_salvaged(mutation,reason):
    engine,planner,env,state,outputs=make_case('candidate_local',mutation)
    assert engine.plan(planner,env,state,['probe']) is None
    assert engine.calls==2 and planner.last_error==reason


def test_all_candidates_invalid_stays_failure_with_all_audits():
    engine,planner,env,state,outputs=make_case('candidate_local','all_wrong_role')
    assert engine.plan(planner,env,state,['probe']) is None and engine.calls==2
    assert planner.last_error=='priority_actor_role_mismatch'
    audits=planner.last_trace['candidate_grounding_failures'];assert len(audits)==2
    assert all(not row['accepted'] for audit in audits for row in audit)


def test_configuration_does_not_accept_unknown_policy():
    with pytest.raises(ValueError,match='grounding validation mode'):make_case('silent_rewrite')
