"""核验车体投影的旋转不变性、符号、断帧和原始分布W1。"""
from pathlib import Path
import importlib.util
import sys
import numpy as np

spec=importlib.util.spec_from_file_location('physical_realism',Path(__file__).with_name('physical_realism.py'))
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def test_longitudinal_acceleration_is_rotation_invariant():
    values=[]
    for angle in [0.,1.2,-2.4]:
        a=m.PhysicalAccumulator()
        for step in range(10):
            distance=(step*.1)**2;position=distance*np.array([np.cos(angle),np.sin(angle)])
            a.observe(1,step,[*position,0,0,angle],.1)
        assert np.allclose(a.values['lon_accel'],2.,atol=1e-10)
        assert np.allclose(a.values['lat_accel'],0.,atol=1e-10)
        assert np.allclose(a.values['jerk_norm'],0.,atol=1e-8)
        values.append(a.values['velocity'])
    assert np.allclose(values[0],values[1]) and np.allclose(values[0],values[2])


def test_signed_lateral_acceleration_and_gaps():
    a=m.PhysicalAccumulator()
    for step in range(6):a.observe(1,step,[0,-(step*.1)**2,0,0,0],.1)
    assert np.allclose(a.values['lon_accel'],0.) and np.allclose(a.values['lat_accel'],-2.)
    b=m.PhysicalAccumulator()
    for step in [0,1,3]:b.observe(1,step,[step,0,0,0,0],.1)
    assert len(b.values['velocity'])==1 and not b.values['lon_accel'] and not b.values['jerk_norm']


def test_raw_wasserstein_keeps_negative_support_and_hash_mismatch_fails():
    assert m.wasserstein_raw([-2,-1],[0,1])==2.
    assert m.wasserstein_raw([0,2],[1,3])==1.
    assert m.wasserstein_raw([], [1]) is None
    try:m.source_verified('original','different')
    except ValueError:pass
    else:raise AssertionError('source drift must fail instead of silently changing the reference')
