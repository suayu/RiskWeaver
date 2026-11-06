"""确保日志不会跨场景/重复源帧硬配对，并核对真正时间对齐的距离。"""
import pytest
from audit import segments,unique_segment,aligned_distances,interaction_threshold


def error(text='bad'):
    return '[大模型规划器] 本次高级攻击规划失败，已跳过且仿真继续：'+text


def test_ordered_scene_step_association_and_no_cross_scene_carryover():
    lines=['Initial data saved to p/scenario_0000_initial.json','[Adversarial Generator] Planning attack at step 0',error('first'),
        'Initial data saved to p/scenario_0001_initial.json',error('orphan'),
        '[Adversarial Generator] Planning attack at step 11',error('second')]
    p=segments('\n'.join(lines),[53,140])
    assert p['initialization_order_verified'] and len(p['unassociated_failures'])==1
    assert unique_segment(p,53,0)['failure_lines'][0]['text']==error('first')
    assert unique_segment(p,140,11)['failure_lines'][0]['text']==error('second')
    assert unique_segment(p,140,0) is None


@pytest.mark.parametrize('mode',['duplicate_plan','multiple_errors','bad_scene_order'])
def test_ambiguous_associations_stay_unknown(mode):
    lines=['Initial data saved to p/scenario_0000_initial.json','[Adversarial Generator] Planning attack at step 0',error()]
    if mode=='duplicate_plan':lines+=['[Adversarial Generator] Planning attack at step 0',error()]
    if mode=='multiple_errors':lines+=[error('other')]
    if mode=='bad_scene_order':lines[0]='Initial data saved to p/scenario_0001_initial.json'
    assert unique_segment(segments('\n'.join(lines),[53,140]),53,0) is None


def test_distance_uses_each_corresponding_second_not_all_against_t3():
    distances,future=aligned_distances(dict(anchors=[[0,0],[0,10],[0,20],[0,30]]),dict(ego_state=[0,0,0,10,0]))
    assert distances==[0.,0.,0.] and future==[[0,10],[0,20],[0,30]]


def test_threshold_must_come_from_unique_frozen_condition():
    assert interaction_threshold('if closest_approach > 15.0:\n raise ValueError()')==15.
    with pytest.raises(ValueError,match='ambiguous'):interaction_threshold('if something > 15:\n pass')
    with pytest.raises(ValueError,match='ambiguous'):interaction_threshold('if closest_approach > 15:\n pass\nif closest_approach > 9:\n pass')
