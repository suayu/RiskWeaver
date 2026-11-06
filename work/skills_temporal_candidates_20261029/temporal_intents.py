"""稀疏意图的帧间交互和速度对齐延续；不输出可直接执行的连续控制。"""
import hashlib
import inspect
import ast
import textwrap
import numpy as np

# 此AST指纹在项目固定Python环境中与295源码冻结快照逐节点核对；升级解释器也需重新审查。
VALIDATOR_AST_SHA256='8fe2525715a4ffb8545de86a59f867a1bc8eb98c58eefbee533e3a140ba53214'
DISTANCE_ERROR_PREFIX='锚点与自车三秒预测位置的最近距离为 '


def fractional_minimum(anchors,ego,start=.1,stop=3.):
    """分段直线意图与CV自车的相对二次距离精确最小值；不是执行轨迹或碰撞证书。"""
    a=np.asarray(anchors,float);e=np.asarray(ego,float)
    if a.shape!=(4,2) or e.ndim!=1 or len(e)<4 or not np.isfinite(a).all() or not np.isfinite(e[:4]).all():
        raise ValueError('invalid_fractional_geometry')
    if not 0<start<=stop<=3:raise ValueError('fractional_time_range_invalid')
    candidates=[]
    for segment in range(3):
        lo=max(start,float(segment));hi=min(stop,float(segment+1))
        if hi<lo:continue
        velocity=a[segment+1]-a[segment]-e[2:4]
        relative=a[segment]-e[:2]-segment*e[2:4]
        denominator=float(velocity@velocity)
        tau=float(np.clip(-relative@velocity/denominator if denominator else 0.,lo-segment,hi-segment))
        gap=float(np.linalg.norm(relative+tau*velocity))
        candidates.append((gap,segment+tau))
    gap,time=min(candidates)
    return dict(min_center_gap_m=gap,relative_time_seconds=time,range_seconds=[start,stop],
        basis='exact_piecewise_linear_sparse_intent_vs_source_CV',actual_execution_or_threat_certified=False)


def guarded_fractional_validator(original,events):
    """只替换整数秒交互判据；先执行原全部前置检查，并复核其唯一剩余cut_in检查。"""
    # 源码守卫防止原方法未来新增检查后，适配器误跳过它们。额度注释不改变方法AST。
    source=textwrap.dedent(inspect.getsource(original));tree=ast.parse(source)
    functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)]
    digest=hashlib.sha256(ast.dump(functions[0],include_attributes=False).encode()).hexdigest() if len(functions)==1 else None
    if digest!=VALIDATOR_AST_SHA256:raise ValueError('original_validator_changed_fractional_adapter_requires_review')
    def validate(plan,state):
        try:return original(plan,state)
        except ValueError as error:
            message=str(error)
            if not (message.startswith(DISTANCE_ERROR_PREFIX) and message.endswith('米，不能形成有效交互')):raise
            result=fractional_minimum(plan['anchors'],state['ego_state'])
            if result['min_center_gap_m']>15.:
                events.append(dict(original_rejection=message,fractional_result=result,threshold_m=15.,accepted=False))
                raise ValueError(f"帧间稀疏意图与自车CV最近中心距{result['min_center_gap_m']:.3f}米，时刻{result['relative_time_seconds']:.3f}秒，超过15米交互门槛") from error
            # 原方法在交互判据之后只剩这一项；不能因为帧间交互成立就跳过终点走廊要求。
            if plan.get('strategy')=='cut_in' and abs(np.asarray(plan['anchors'])[-1,0])>6.5:
                raise ValueError('cut_in 终点没有进入自车行驶走廊')
            events.append(dict(original_rejection=message,fractional_result=result,threshold_m=15.,
                accepted=True,other_original_checks_preserved=True,not_background_or_dynamics_relaxation=True))
            return None
    return validate


def velocity_aligned_candidates(previous,actor,step,dt=.1):
    """延续已接收意图的二阶稀疏运动特征，从实测位置和速度重新实例化。"""
    a=np.asarray(previous['anchors'],float);s=np.asarray(actor,float)
    if a.shape!=(4,2) or s.ndim!=1 or len(s)<4 or not np.isfinite(a).all() or not np.isfinite(s[:4]).all():
        return [],dict(available=False,reason='invalid_prior_or_current_state')
    acceleration=np.diff(a,n=2,axis=0);residual=float(np.linalg.norm(acceleration[1]-acceleration[0]))
    if residual>1e-6:return [],dict(available=False,reason='prior_not_constant_acceleration_sparse_recipe',fit_residual_m=residual)
    old_a=acceleration.mean(axis=0);norm=float(np.linalg.norm(old_a))
    if norm>3.8+1e-8:return [],dict(available=False,reason='prior_acceleration_outside_existing_recipe_cap',norm_mps2=norm)
    if norm<1e-8:return [],dict(available=False,reason='prior_has_no_acceleration_signature')
    if norm>3.8:old_a=old_a*3.8/norm
    elapsed=(step-previous['source_step'])*dt
    if not 0<elapsed<3:return [],dict(available=False,reason='prior_time_not_forward_or_expired')
    old_v0=a[1]-a[0]-.5*old_a;old_current_velocity=old_v0+elapsed*old_a
    t=np.arange(4.)[:,None];output=[]
    for scale in [1.,.5]:
        sparse=s[:2]+t*s[2:4]+.5*t*t*(scale*old_a)
        output.append(dict(target_id=previous['target_id'],strategy=previous['strategy'],attack_family='ongoing_conflict',
            anchors=sparse.tolist(),duration=10,recipe='current_state_velocity_aligned_prior_acceleration_v1',
            prior_episode_id=previous['episode_id'],prior_source_step=previous['source_step'],acceleration_scale=scale,
            prior_sparse_acceleration_mps2=old_a.tolist(),source_velocity_difference_from_quadratic_prior_mps=(s[2:4]-old_current_velocity).tolist(),
            new_tail_is_explicit_model_extension_not_old_forecast=True,continuous_execution_approved=False))
    return output,dict(available=True,source_velocity_difference_mps=(s[2:4]-old_current_velocity).tolist(),
        prior_fit_residual_m=residual,scope='sparse_quadratic_intent_signature_not_old_continuous_trajectory_or_actual_acceleration')
