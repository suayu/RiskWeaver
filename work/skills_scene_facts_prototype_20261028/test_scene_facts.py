"""位置、航向、陈旧身份和原验证保留的回归；不调用外部模型。"""
import copy
import math
from pathlib import Path
import sys

import pytest
from scene_facts import relation_card,build_cards,validate_claims,FIELDS


def fixture():
    state=dict(ego_state=[0.,0.,0.,1.,math.pi/2],agents=[dict(id=0,state=[.06,-24.86,0.,1.,math.pi/2])])
    candidates=[dict(candidate_id='a0',target_id=0)]
    cards=build_cards(state,candidates,0);card=cards['a0']
    response=dict(ranking=['a0'],grounding={'a0':dict(scene_fact_id=card['fact_id'],source_step=0,
        longitudinal_relation=card['longitudinal_relation'],lateral_relation=card['lateral_relation'],heading_relation=card['heading_relation'])})
    return state,candidates,cards,response


def test_historical_rear_vehicle_is_not_ahead_left():
    _,_,cards,response=fixture();c=cards['a0']
    assert c['longitudinal_relation']=='behind' and c['lateral_relation']=='aligned'
    validate_claims(response,cards)
    response['grounding']['a0']['longitudinal_relation']='ahead'
    with pytest.raises(ValueError,match='longitudinal_relation'):validate_claims(response,cards)


def test_historical_crossing_not_oncoming():
    ego=[0,0,0,1,math.pi/2];actor=[7,26,1,0,0]
    c=relation_card(ego,actor,1,44)
    assert c['lateral_relation']=='right' and c['longitudinal_relation']=='ahead'
    assert c['heading_relation']=='crossing_headings' and c['heading_difference_degrees']==90
    response=dict(ranking=['a1'],grounding={'a1':dict(scene_fact_id=c['fact_id'],source_step=44,
        longitudinal_relation='ahead',lateral_relation='right',heading_relation='opposing_headings')})
    with pytest.raises(ValueError,match='heading_relation'):validate_claims(response,{'a1':c})


@pytest.mark.parametrize('angle,expected',[(0.,'similar_headings'),(.699,'similar_headings'),(.7,'crossing_headings'),(2.4,'crossing_headings'),(2.401,'opposing_headings'),(math.pi,'opposing_headings')])
def test_existing_recipe_heading_boundaries(angle,expected):
    c=relation_card([0,0,0,0,0],[1,0,0,0,angle],0,0)
    assert c['heading_relation']==expected


def test_rigid_transform_invariant_and_input_not_modified():
    ego=[2.,3.,0.,1.,1.1];actor=[4.,-5.,0.,1.,-.4];prior=copy.deepcopy((ego,actor))
    first=relation_card(ego,actor,1,11);angle=.73
    def transform(x):
        return [x[0]*math.cos(angle)-x[1]*math.sin(angle)+12,x[0]*math.sin(angle)+x[1]*math.cos(angle)-4,0,1,x[4]+angle]
    after=relation_card(transform(ego),transform(actor),1,11)
    for k in ['right_m','forward_m','heading_difference_degrees']:assert math.isclose(first[k],after[k],abs_tol=1e-10)
    assert all(first[k]==after[k] for k in ['longitudinal_relation','lateral_relation','heading_relation'])
    assert prior==(ego,actor)


@pytest.mark.parametrize('field,value',[('source_step',1),('source_step',False),('scene_fact_id','actor1@step0:pose'),('lateral_relation','left')])
def test_wrong_time_identity_or_side_is_rejected(field,value):
    _,_,cards,response=fixture();response['grounding']['a0'][field]=value
    with pytest.raises(ValueError):validate_claims(response,cards)


def test_missing_claim_and_unknown_candidate_never_pass():
    _,_,cards,response=fixture();del response['grounding']['a0']['heading_relation']
    with pytest.raises(ValueError):validate_claims(response,cards)
    _,_,cards,response=fixture();response['ranking']=['bad'];response['grounding']={'bad':response['grounding']['a0']}
    with pytest.raises(ValueError):validate_claims(response,cards)


def test_duplicate_actor_nonfinite_and_inactive_identity_not_in_snapshot():
    state,candidates,_,_=fixture();state['agents']*=2
    with pytest.raises(ValueError):build_cards(state,candidates,0)
    state,candidates,_,_=fixture();state['ego_state'][0]=float('nan')
    with pytest.raises(ValueError):build_cards(state,candidates,0)
    state,candidates,_,_=fixture();state['agents']=[]
    with pytest.raises(ValueError):build_cards(state,candidates,0)


def test_claim_validation_does_not_certify_free_prose():
    _,_,cards,response=fixture();response['scene_analysis']='incorrect free prose retained for audit'
    before=copy.deepcopy(response);validate_claims(response,cards)
    assert response==before


def test_old_validator_still_rejects_wrong_target_and_horizon():
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'work/skills_incremental_evidence_20261025'))
    from test_incremental_planner import fixture as old_fixture
    from fact_planner import SceneFactPlanner
    old,catalog,response=old_fixture()
    _,_,cards,_=fixture()
    engine=SceneFactPlanner.__new__(SceneFactPlanner)
    engine.fact_grounding=True;engine.incremental_cards=old.incremental_cards;engine.priority_evidence=old.priority_evidence
    engine.scene_fact_cards={'x0':cards['a0']};c=cards['a0']
    response['grounding']['x0'].update(scene_fact_id=c['fact_id'],source_step=0,
        longitudinal_relation=c['longitudinal_relation'],lateral_relation=c['lateral_relation'],heading_relation=c['heading_relation'])
    engine.validate_response(response,'r','s','c',catalog,False)
    for key,value in [('target_id',9),('active_horizon_seconds',3.)]:
        invalid=copy.deepcopy(response);invalid['grounding']['x0'][key]=value
        with pytest.raises(ValueError):engine.validate_response(invalid,'r','s','c',catalog,False)
    assert FIELDS<=set(response['grounding']['x0'])
