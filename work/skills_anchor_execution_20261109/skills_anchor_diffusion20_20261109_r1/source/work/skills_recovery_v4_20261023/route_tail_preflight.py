"""可选的路线尾部观测余量筛查；不修改可达算法、真实路线或D窗口。"""
import copy
import math
from types import SimpleNamespace as NS
import numpy as np
from policies.risk_metrics import RouteReachability
from reachability_ready_planner import ReachabilityReadyPlanner


def route_tail_readiness(env,config,duration_frames=10):
    """以沿路线匀速前进的名义状态检查攻击有效期末，不能作为未来安全保证。"""
    if isinstance(duration_frames,bool) or not isinstance(duration_frames,int) or not 1<=duration_frames<=100:raise ValueError('invalid duration')
    if str(getattr(config,'backend','route_grid'))!='route_grid':raise ValueError('route_tail_backend_not_supported')
    reach=RouteReachability(config);route,arc,tangent=reach._route_geometry(env.scenario_dict['route'])
    ego=np.asarray(env.ego_state,float);nearest=int(np.argmin(np.linalg.norm(route-ego[:2],axis=1)))
    direction=tangent[min(nearest,len(tangent)-1)];normal=np.array([-direction[1],direction[0]])
    s=float(arc[nearest]);d=float((ego[:2]-route[nearest])@normal)
    speed=max(0.,float(ego[2:4]@direction));duration=duration_frames*float(env.dt);future_s=s+speed*duration
    common=dict(duration_frames=duration_frames,lookahead_seconds=duration,remaining_route_m=float(arc[-1]-s),
        nominal_longitudinal_speed_mps=speed,unchanged_evaluation_horizon_seconds=reach.horizon,
        assumption='route_following_constant_longitudinal_speed_without_traffic',
        scope='planning-opportunity filter only; no future-ground-truth, no road extension, no execution approval')
    if future_s>=arc[-1]:return dict(common,ready=False,reason='nominal_route_end_within_attack_duration',future_unobstructed_area_m2=0.)
    center,direction=reach._sample_route(route,arc,tangent,future_s);normal=np.array([-direction[1],direction[0]])
    predicted=ego.copy();predicted[:2]=center+normal*d;predicted[2:4]=direction*speed;predicted[4]=math.atan2(direction[1],direction[0])
    probe=NS(ego_state=predicted,data_dict=env.data_dict,agent_active=np.zeros_like(env.agent_active,dtype=bool),
        pending_joint_trajectory=None,scenario_dict=env.scenario_dict,dt=env.dt)
    result=reach.compute(probe,use_joint_trajectory=False,static_obstacles=[])
    area=float(result['area_m2']);ready=math.isfinite(area) and area>0 and bool(result['solvable'])
    return dict(common,ready=ready,reason='ready' if ready else 'route_tail_measurement_margin_insufficient',
                future_unobstructed_area_m2=area,area_curve=result['area_curve'])


class RouteTailReadyPlanner(ReachabilityReadyPlanner):
    def build_context(self,env,state,planner,profile_context=None):
        context=super().build_context(env,state,planner,profile_context)
        if not context['candidates'] and not context.get('obstacle_candidates'):return context
        result=route_tail_readiness(env,env.cfg.sim.evaluation.reachability,10)
        context['route_tail_preflight']=result
        if not result['ready']:
            # 完整目录留在审计字段中，不能把有候选但暂缓的事件写成无候选。
            context['route_tail_withheld_candidates']=copy.deepcopy(context['candidates'])
            context['route_tail_withheld_obstacles']=copy.deepcopy(context.get('obstacle_candidates',[]))
            context['candidates']=[]
            if 'obstacle_candidates' in context:context['obstacle_candidates']=[]
        return context

    def plan(self,planner,*args,**kwargs):
        result=super().plan(planner,*args,**kwargs)
        check=planner.last_trace.get('context',{}).get('route_tail_preflight',{})
        if check.get('ready') is False:
            planner.last_trace.update(stage='route_tail_preflight_decline',plan_validation='route_tail_measurement_margin_insufficient')
            if result is not None:result['reason']=check['reason']
        return result
