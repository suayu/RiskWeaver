"""v2可见视图不能改变全精度锚点、候选ID或安全条件。"""
import json
from types import SimpleNamespace as NS
import numpy as np
from test_ego_profile import environment,planner as fixture_planner
from compact_planner import CompactSkillsPlanner,compact_prompt


def test_display_rounding_does_not_change_compiled_geometry():
    env=environment();p=fixture_planner();p.use_obstacles=False;p.provider='dashscope'
    engine=CompactSkillsPlanner(NS(llm=NS(skills=NS(level='none',candidate_expansion_enabled=True,max_repairs=0)),ego_profile=NS()),1)
    expected=engine.build_context(env,env.get_state_for_planning(),p)
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.split('\n',1)[1]);assert 'semantic_precheck' not in payload['context']
        assert payload['display_contract']['precision_decimals']==3
        cid=payload['context']['candidates'][0]['candidate_id']
        response={k:payload[k] for k in ('request_id','state_hash','catalog_hash')}
        response.update(ranking=[cid],scene_analysis='test',candidate_reasons={cid:'minimum aligned distance'},no_attack_reason='')
        return '',response
    p._request_attack_plan=request
    result=engine.plan(p,env,env.get_state_for_planning(),[])
    candidate=next(c for c in expected['candidates'] if c['candidate_id']==result['candidate_id'])
    np.testing.assert_array_equal(result['anchors'],candidate['anchors'])
    assert p._request_attack_plan is request


def test_empty_background_margin_means_no_other_object_not_unchecked():
    env=environment();p=fixture_planner();p.use_obstacles=False
    engine=CompactSkillsPlanner(NS(llm=NS(skills=NS(candidate_expansion_enabled=True)),ego_profile=NS()))
    context=engine.build_context(env,env.get_state_for_planning(),p)
    assert context['candidates']
    for c in context['candidates']:
        evidence=c['background_safety_assessment']
        assert evidence['status']=='no_non_target_objects' and evidence['minimum_sampled_margin_m'] is None
        assert evidence['final_joint_safety_required'] and evidence['threshold_m']==.5


def test_roles_and_relative_features_use_ego_heading():
    env=environment();p=fixture_planner();p.use_obstacles=False
    engine=CompactSkillsPlanner(NS(llm=NS(skills=NS()),ego_profile=NS()))
    context=engine.build_context(env,env.get_state_for_planning(),p)
    feature=context['scene_features'][0]
    assert feature['target_id']==0 and feature['role']=='background_actor_not_ego'
    assert feature['relative_forward_m']==18 and feature['relative_right_m']==0
