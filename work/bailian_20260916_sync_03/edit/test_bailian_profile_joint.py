"""百炼联合模式的候选边界与实际规划协议回归；传输层使用替身。"""
import json

from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.obstacles import ObstacleCatalog
from policies.profile_obstacles import build_obstacle_candidates, chosen_obstacle
from policies.profile_planner import rank_profile_candidates
from test_ego_profile import environment,pipeline,planner as fixture_planner


def obstacle_planner():
    planner=fixture_planner()
    planner.provider='dashscope'
    planner.use_obstacles=True
    planner.obstacle_catalog=ObstacleCatalog()
    planner._validate_obstacle_plan=LLMAdversarialPlanner._validate_obstacle_plan.__get__(planner)
    return planner


def context_for_obstacles(env,planner):
    context=pipeline().context(env,.5)
    context['obstacle_candidates']=build_obstacle_candidates(env,planner,context['candidates'],pipeline().builder)
    return context


def test_joint_obstacle_candidate_with_no_dynamic_attacker():
    env=environment()
    env.data_dict['agent'][-1][0,1]=6.
    planner=obstacle_planner()
    context=context_for_obstacles(env,planner)
    assert not context['candidates']
    assert context['obstacle_candidates']
    candidate=context['obstacle_candidates'][0]
    assert candidate['placement']['type']=='cone_line'
    assert candidate['background_margin_m']>.5

    def response(prompt,scene_image=None):
        payload=json.loads(prompt.split('\n',1)[1])
        return '',dict(request_id=payload['request_id'],ranking=[],no_attack_reason='no dynamic target',
                       obstacle_candidate_id=candidate['obstacle_candidate_id'])
    planner._request_attack_plan=response
    plan=rank_profile_candidates(planner,env.get_state_for_planning(),[],context)
    assert not plan['attack'] and plan['obstacle_plan']==[candidate['placement']]


def test_joint_obstacle_id_and_combination_are_closed_catalog():
    env=environment()
    planner=obstacle_planner()
    context=context_for_obstacles(env,planner)
    assert context['candidates']
    if context['obstacle_candidates']:
        candidate=context['obstacle_candidates'][0]
        assert chosen_obstacle({'obstacle_candidate_id':candidate['obstacle_candidate_id']},context,[])==[candidate['placement']]
    try:
        chosen_obstacle({'obstacle_candidate_id':'invented'},context,[])
    except ValueError:
        pass
    else:
        raise AssertionError('unknown obstacle ID accepted')


def test_trajectory_mode_retains_existing_three_field_ranking():
    env=environment()
    planner=fixture_planner()
    planner.provider='dashscope'
    planner.use_obstacles=False
    context=pipeline().context(env,.5)
    result=rank_profile_candidates(planner,env.get_state_for_planning(),[],context)
    assert result['attack']
    assert 'obstacle_plan' not in result
