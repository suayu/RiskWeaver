"""区分相对危险度测量与实际执行区域；默认完整保留历史判定。"""
import math


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def assess_reachability_execution(event,policy='legacy_relative'):
    """只评估执行面积这一道门；动力学、背景OBB等其他硬门槛仍由原检查器负责。"""
    if policy not in {'legacy_relative','positive_area'}:raise ValueError('unknown execution reachability policy')
    event=event or {};difficulty=event.get('difficulty');baseline=event.get('original_area_m2');actual=event.get('dangerous_area_m2')
    relative_valid=_finite(difficulty) and _finite(baseline) and baseline>0
    if policy=='legacy_relative':
        # 与原入口相同：相对指标有限且危险未来可解。新模式必须显式打开。
        allowed=_finite(difficulty) and bool(event.get('dangerous_solvable'))
        reason='legacy_gate_passed' if allowed else 'legacy_relative_measurement_or_solvability_failed'
    elif not _finite(actual) or actual<=0:
        allowed=False;reason='actual_execution_area_nonpositive_or_invalid'
    elif not bool(event.get('dangerous_solvable')):
        allowed=False;reason='actual_execution_future_unsolvable'
    else:
        allowed=True;reason='positive_actual_area' if relative_valid else 'positive_actual_area_relative_measurement_invalid'
    return dict(allowed=bool(allowed),policy=policy,reason=reason,relative_measurement_valid=bool(relative_valid),
        original_area_m2=baseline if _finite(baseline) else None,
        dangerous_area_m2=actual if _finite(actual) else None,
        relative_difficulty=difficulty if _finite(difficulty) else None,
        invalid_D_policy='retain_invalid_window; D remains null when no valid windows; hold controller intensity')
