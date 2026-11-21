"""低成本机会监测；只决定何时查询，不能替代候选、道路或联合安全验证。"""
import math


class OpportunityScheduler:
    def __init__(self,cfg=None,dt=.1):
        self.dt=float(dt)
        self.minimum=int(getattr(cfg,'min_interval_frames',5))
        self.heartbeat=int(getattr(cfg,'heartbeat_frames',20))
        self.confirm=int(getattr(cfg,'confirmation_frames',2))
        self.horizon=float(getattr(cfg,'horizon_seconds',5.))
        self.radius=float(getattr(cfg,'interaction_distance_m',15.))
        self.urgent=float(getattr(cfg,'urgent_distance_m',5.))
        self.reserve_until=int(getattr(cfg,'reserve_until_frame',50))
        if self.dt<=0 or min(self.minimum,self.heartbeat,self.confirm)<1 or self.horizon<=0 or self.radius<=0:
            raise ValueError('invalid_event_trigger_settings')
        self.reset()

    def reset(self):
        self.last_query=None;self.last_signature=None;self.pending=None;self.stable=0

    def decide(self,state,step,remaining,active_intent=None):
        """用当前位置速度计算CV最近点，不读取未来回放或额外采样。"""
        ego=state['ego_state'];actors=state.get('agents',[]);opportunities=[];imminent=False
        ids={a['id'] for a in actors}
        for actor in actors:
            a=actor['state'];dx=a[0]-ego[0];dy=a[1]-ego[1];vx=a[2]-ego[2];vy=a[3]-ego[3]
            speed2=vx*vx+vy*vy
            time=max(0.,min(self.horizon,-(dx*vx+dy*vy)/speed2)) if speed2>1e-12 else 0.
            gap=math.hypot(dx+time*vx,dy+time*vy)
            if gap<=self.radius:
                opportunities.append(actor['id'])
                imminent|=gap<=self.urgent and time<=1.5
        route=state.get('route',[])
        static_possible=any(math.hypot(p[0]-ego[0],p[1]-ego[1])>=15 for p in route)
        # 已经在自车前方的静态攻击继续存在时，不重复创建同类阻挡。
        ahead_static=False
        for obstacle in state.get('static_obstacles',[]):
            center=obstacle.get('center')
            if center and (center[0]-ego[0])*math.cos(ego[4])+(center[1]-ego[1])*math.sin(ego[4])>=0:
                ahead_static=True
        signature=('dynamic',tuple(sorted(opportunities))) if opportunities else ('static',) if static_possible and not ahead_static else None
        self.stable=self.stable+1 if signature==self.pending else 1
        self.pending=signature
        active_target=(active_intent or {}).get('target_id')
        target_lost=bool(active_intent) and active_target not in ids
        result=dict(query=False,reason='no_usable_observed_opportunity',opportunity=signature,
                    candidate_actor_ids=opportunities,imminent_CV_proxy=imminent,
                    remaining_physical_or_logical_budget=remaining)
        if remaining<=0:
            result['reason']='budget_exhausted';return result
        if active_intent and not target_lost:
            result['reason']='active_intent_retained';return result
        if signature is None:
            return result
        elapsed=step-self.last_query if self.last_query is not None else None
        if elapsed is not None and elapsed<self.minimum and not (target_lost or imminent):
            result['reason']='cooldown';return result
        if remaining==1 and step<self.reserve_until and not (target_lost or imminent):
            result['reason']='reserve_for_later_opportunity';return result
        if self.last_query is None:
            due=True;reason='initial_opportunity'
        elif target_lost:
            due=True;reason='active_target_lost'
        elif signature!=self.last_signature and self.stable>=self.confirm:
            due=True;reason='changed_interaction_opportunity'
        elif imminent and elapsed>=self.minimum:
            due=True;reason='imminent_interaction_proxy'
        else:
            due=elapsed>=self.heartbeat;reason='opportunity_heartbeat' if due else 'unchanged_opportunity'
        result.update(query=due,reason=reason)
        return result

    def record_query(self,step,decision):
        self.last_query=int(step);self.last_signature=decision['opportunity']
