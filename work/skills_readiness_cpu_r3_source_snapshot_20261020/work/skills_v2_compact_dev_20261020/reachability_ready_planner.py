"""先确认原始面积分母可测，再请求新攻击；不改可达算法或执行门槛。"""
import copy
import math

from compact_planner import CompactSkillsPlanner
from policies.risk_metrics import RouteReachability


def baseline_readiness(env, config):
    """使用相同route_grid与CV原场景，明确区分测量失效和攻击危险。"""
    if str(getattr(config, 'backend', 'route_grid')) != 'route_grid':
        raise ValueError('development preflight supports the frozen route_grid backend only')
    baseline = RouteReachability(config).compute(env, use_joint_trajectory=False,
                                               static_obstacles=env.get_static_obstacles())
    area = float(baseline['area_m2'])
    ready = math.isfinite(area) and area > 0 and bool(baseline['solvable'])
    return dict(ready=ready, reason='ready' if ready else 'baseline_reachability_not_ready',
                baseline_area_m2=area, baseline_solvable=bool(baseline['solvable']),
                area_curve=baseline.get('area_curve', []), backend=baseline.get('backend'),
                horizon_seconds=baseline.get('horizon_seconds'),
                scope='current pre-attack measurement readiness; no assertion of attack safety')


class ReachabilityReadyPlanner(CompactSkillsPlanner):
    def build_context(self, env, state, planner, profile_context=None):
        context = super().build_context(env, state, planner, profile_context)
        if not bool(getattr(self.settings, 'reachability_preflight_enabled', False)):
            return context
        if not context['candidates'] and not context.get('obstacle_candidates'):
            context['measurement_preflight'] = dict(ready=None, reason='not_evaluated_empty_catalog')
            return context
        result = baseline_readiness(env, env.cfg.sim.evaluation.reachability)
        context['measurement_preflight'] = result
        if not result['ready']:
            # 被暂缓的完整候选留在trace，不能伪装成模板没有覆盖到该场景。
            context['preflight_withheld_candidates'] = copy.deepcopy(context['candidates'])
            context['preflight_withheld_obstacles'] = copy.deepcopy(context.get('obstacle_candidates', []))
            context['candidates'] = []
            if 'obstacle_candidates' in context:
                context['obstacle_candidates'] = []
        return context

    def plan(self, planner, *args, **kwargs):
        output = super().plan(planner, *args, **kwargs)
        preflight = planner.last_trace.get('context', {}).get('measurement_preflight', {})
        if preflight.get('ready') is False:
            planner.last_trace.update(stage='preflight_decline',
                                      plan_validation='baseline_measurement_not_ready')
            if output is not None:
                output['reason'] = 'baseline_reachability_not_ready'
        return output
