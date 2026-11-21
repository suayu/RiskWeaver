"""离线未来事件时刻诊断；只构造稀疏锚点，不生成连续执行控制。"""
import math
import numpy as np


def solve_sparse_event(ego,actor,time,cap=3.8):
    """按四个整数秒锚点的线性插值反解系数；不把二次模型时刻当作插值到达时刻。"""
    e,s=np.asarray(ego,float),np.asarray(actor,float)
    if e.ndim!=1 or s.ndim!=1 or min(e.size,s.size)<4 or not np.isfinite(e[:4]).all() or not np.isfinite(s[:4]).all():
        raise ValueError('invalid_source_state')
    if not math.isfinite(time) or not 0<time<=3 or not math.isfinite(cap) or not 0<cap<=3.8:
        raise ValueError('invalid_time_or_original_cap')
    f=time-math.floor(time);basis=time*time+f*(1-f)
    point=e[:2]+time*e[2:4]
    acceleration=2*(point-s[:2]-time*s[2:4])/basis
    norm=float(np.linalg.norm(acceleration))
    if norm>cap:return dict(feasible=False,reason='required_sparse_acceleration_above_original_cap',
        requested_event_seconds=float(time),required_sparse_acceleration_norm_mps2=norm,cap_mps2=cap)
    t=np.arange(4.)[:,None];anchors=s[:2]+t*s[2:4]+.5*t*t*acceleration
    sampled=np.array([np.interp(time,np.arange(4),anchors[:,i]) for i in range(2)])
    return dict(feasible=True,anchors=anchors.tolist(),requested_event_seconds=float(time),
        desired_ego_CV_position_world=point.tolist(),sampled_sparse_event_position_world=sampled.tolist(),
        sparse_event_residual_m=float(np.linalg.norm(sampled-point)),
        required_sparse_acceleration_norm_mps2=norm,sparse_acceleration_mps2=acceleration.tolist(),cap_mps2=cap,
        quadratic_interpolation_basis_corrected=True,future_is_source_CV_not_actual_policy=True,
        background_dynamics_area_and_diffusion_execution_not_certified=True)
