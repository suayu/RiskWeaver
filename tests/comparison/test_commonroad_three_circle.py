"""实际官方三圆模式必须保留封闭场景为零，同时不把邻车净距误判为碰撞。"""
from pathlib import Path
import pytest
pytest.importorskip('commonroad_reach')
from experiments.comparison.commonroad_worker import compute


def request():
    return {'request_id':1,'route':[[0,0],[100,0]],'ego':[20,0,2,0,0,5.2897,2.3322,1],
            'dt':.5,'steps':6,'lane_half_width':4.,'max_speed':25.,'max_lateral_speed':3.,
            'a_lon':[-4.,2.],'a_lat':[-2.5,2.5],'grid_size':1.,'inflation_mode':3,
            'dynamic':[[] for _ in range(7)],'static':[]}


@pytest.mark.skipif(not Path('/home2/zhaoyx/scenario-dreamer/metadata/commonroad_reach_lut_20261014/lut_wheelbase_35.txt').exists(),reason='official LUT unavailable')
def test_three_circle_distinguishes_neighbor_and_full_blockage():
    q=request();q['dynamic']=[[([20,3.4],0.,4.,2.)] for _ in range(7)]
    neighbor=compute(q)
    assert neighbor['area_m2']>0
    assert neighbor['ego_inflation']=='three_circle_with_official_LUT'
    q['dynamic']=[[([20,0],0.,100.,100.)] for _ in range(7)]
    assert compute(q)['area_m2']==0
