"""事实复核器验证源帧方向、刚体变换和精确引用；不替代语言理解验收。"""
import math
import pytest
from review_accepted_reasons import radial_gap_rate,heading_difference,add_finding


def test_current_separating_behind_vehicle_is_not_closing():
    # 自车向北5m/s，后车仅3m/s，当前距离正在增加2m/s。
    e=[0,0,0,5,math.pi/2];a=[0,-20,0,3,math.pi/2]
    assert radial_gap_rate(e,a)==2.
    a[3]=7.;assert radial_gap_rate(e,a)==-2.


def test_radial_rate_invariant_to_translation_rotation():
    ego=[1,2,3,4,.3];actor=[-4,9,8,-2,1.2]
    def change(a,angle):
        c,s=math.cos(angle),math.sin(angle)
        return [c*a[0]-s*a[1]+30,s*a[0]+c*a[1]-12,c*a[2]-s*a[3],s*a[2]+c*a[3],a[4]+angle]
    assert math.isclose(radial_gap_rate(ego,actor),radial_gap_rate(change(ego,1.7),change(actor,1.7)),abs_tol=1e-12)
    assert math.isclose(heading_difference(ego,actor),heading_difference(change(ego,1.7),change(actor,1.7)),abs_tol=1e-12)


@pytest.mark.parametrize('actor',[[0,0,0,1,0],[float('nan'),1,0,1,0]])
def test_undefined_or_missing_geometry_never_filled_zero(actor):
    with pytest.raises(ValueError):radial_gap_rate([0,0,0,1,0],actor)


def test_review_cannot_invent_quote_or_candidate_reason():
    r=dict(review_id='x',scenario_index=1,source_step=0,scene_analysis='source text',all_ranked_reasons={'a':'ranked reason'})
    out=[]
    with pytest.raises(ValueError,match='quote_not_in_original'):add_finding(out,r,'scene_analysis','invented','fact',{},'no')
    assert out==[]
    add_finding(out,r,'candidate_reasons','ranked reason','fact',{'value':1},'explanation',cid='a')
    assert out[0]['quote']=='ranked reason' and 'not_a_general' in out[0]['scope']
