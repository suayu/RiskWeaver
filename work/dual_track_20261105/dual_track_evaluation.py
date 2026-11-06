"""双轨评价：固定有限参考控制库，不读取碰撞或路线偏离作为挑战输入。"""
from itertools import product
import hashlib,json
import numpy as np

VERSION='reference_library_dual_v1'


def ability_score(challenge,completed,collision,off_route,progress,alpha=.5):
    # 结果只影响任务得分一次，不再进入挑战度或额外乘碰撞惩罚。
    if challenge is None:return None
    if not np.isfinite(challenge) or not 0<=challenge<=1:raise ValueError('invalid_challenge')
    if any(v is None for v in [completed,collision,off_route,progress]):return None
    success=bool(completed and not collision and not off_route)
    return 100*challenge*(1. if success else alpha*float(np.clip(progress,0,1)))


def outcome_score(result):
    if not result.get('executed_steps') or any(result.get(k) is None for k in ['collision','off_route','completed','progress']):return None
    return .5*bool(result['collision'])+.2*bool(result['off_route'])+.2*(1-bool(result['completed']))+.1*(1-float(np.clip(result['progress'],0,1)))


def overlap(centers,yaws,ego_size,others,other_yaws,sizes):
    if not len(others):return np.zeros(len(centers),bool)
    delta=others[None,:,:]-centers[:,None,:]
    ea=np.stack((np.cos(yaws),np.sin(yaws)),axis=1);el=np.stack((-np.sin(yaws),np.cos(yaws)),axis=1)
    oa=np.stack((np.cos(other_yaws),np.sin(other_yaws)),axis=1);ol=np.stack((-np.sin(other_yaws),np.cos(other_yaws)),axis=1)
    hit=np.ones(delta.shape[:2],bool)
    axes=[np.broadcast_to(ea[:,None,:],delta.shape),np.broadcast_to(el[:,None,:],delta.shape),np.broadcast_to(oa[None,:,:],delta.shape),np.broadcast_to(ol[None,:,:],delta.shape)]
    for axis in axes:
        radius=(ego_size[0]/2*np.abs(np.sum(axis*ea[:,None,:],axis=2))+ego_size[1]/2*np.abs(np.sum(axis*el[:,None,:],axis=2))+
                sizes[None,:,0]/2*np.abs(np.sum(axis*oa[None,:,:],axis=2))+sizes[None,:,1]/2*np.abs(np.sum(axis*ol[None,:,:],axis=2)))
        hit&=np.abs(np.sum(delta*axis,axis=2))<=radius
    return hit.any(axis=1)


class ReferenceChallenge:
    def __init__(self,horizon=3.,dt=.1,lane_half_width=4.,progress_ratio=.2):
        self.horizon=horizon;self.dt=dt;self.half_width=lane_half_width;self.progress_ratio=progress_ratio
        controls=list(product([-4.,-2.,0.,2.],[-2.5,0.,2.5]));self.library=np.asarray(list(product(controls,repeat=2)),float)
        if horizon<=0 or dt<=0 or not 0<=progress_ratio<=1:raise ValueError('invalid_reference_protocol')

    def _evaluate(self,snapshot,shift=0.,delay=0.,scale=1.):
        route=np.asarray(snapshot['route'],float);ego=np.asarray(snapshot['ego'],float);agents=np.asarray(snapshot['agents'],float);active=np.asarray(snapshot['active'],bool)
        if route.ndim!=2 or len(route)<2 or ego.size<7 or not np.isfinite(ego[:7]).all():raise ValueError('incomplete_reference_state')
        if agents.size==0:agents=agents.reshape(0,8)
        if ego[5]<=0 or ego[6]<=0 or not np.isfinite(route).all():raise ValueError('invalid_reference_geometry')
        good=np.r_[True,np.linalg.norm(np.diff(route[:,:2],axis=0),axis=1)>1e-6];route=route[good,:2]
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
        tangent=np.arctan2(np.gradient(route[:,1]),np.gradient(route[:,0]));near=int(np.argmin(np.linalg.norm(route-ego[:2],axis=1)));yaw=tangent[near]
        initial_s=arc[near];initial_d=float((ego[:2]-route[near])@np.array([-np.sin(yaw),np.cos(yaw)]))
        initial_v=float(ego[2]*np.cos(yaw)+ego[3]*np.sin(yaw));initial_vd=float(-ego[2]*np.sin(yaw)+ego[3]*np.cos(yaw))
        n=len(self.library);s=np.full(n,initial_s);d=np.full(n,initial_d);v=np.full(n,initial_v);vd=np.full(n,initial_vd);road=np.ones(n,bool);safe=np.ones(n,bool);cost=np.zeros(n);previous=np.zeros((n,2))
        if agents.ndim!=2 or agents.shape[1]<7 or len(active)!=len(agents):raise ValueError('invalid_agent_reference')
        if not np.isfinite(agents[active,:7]).all() or np.any(agents[active,5:7]<=0):raise ValueError('invalid_active_geometry')
        agents=agents[active];static=snapshot.get('static_obstacles',[])
        for obj in static:
            if not np.isfinite([*obj['center'][:2],obj['yaw'],obj['length'],obj['width']]).all() or min(obj['length'],obj['width'])<=0:raise ValueError('invalid_static_geometry')
        sizes=np.concatenate([agents[:,5:7],np.array([[o['length'],o['width']] for o in static]).reshape(-1,2)])
        oy=np.r_[agents[:,4],[o['yaw'] for o in static]]
        steps=int(round(self.horizon/self.dt))
        for step in range(steps):
            t=(step+1)*self.dt;phase=int(step>=steps//2);u=self.library[:,phase,:]*scale
            if t<=delay:u=np.zeros_like(u)
            nv=np.clip(v+u[:,0]*self.dt,0,25);nvd=np.clip(vd+u[:,1]*self.dt,-3,3)
            s+=(v+nv)*.5*self.dt;d+=(vd+nvd)*.5*self.dt;v,vd=nv,nvd
            road&=(s>=0)&(s<=arc[-1]+1e-8)&(np.abs(d)+ego[6]/2<=self.half_width)
            heading=np.interp(s,arc,np.unwrap(tangent));centers=np.c_[np.interp(s,arc,route[:,0])-np.sin(heading)*d,np.interp(s,arc,route[:,1])+np.cos(heading)*d]
            dynamic=agents[:,:2]+agents[:,2:4]*t
            # 使用场景提供的固定背景未来；不调用被测策略，也不读取其实际碰撞结果。
            oy_current=oy.copy()
            joint=snapshot.get('joint')
            if joint:
                ids={int(a):j for j,a in enumerate(joint['ids'])};original=np.flatnonzero(active)
                for j,aid in enumerate(original):
                    if aid in ids:
                        k=min(max(round(t/snapshot.get('dt',.1))-1,0),len(joint['positions'][ids[aid]])-1)
                        if joint['valid'][ids[aid]][k]:
                            dynamic[j]=joint['positions'][ids[aid]][k]
                            if joint.get('yaws') is not None:oy_current[j]=float(np.asarray(joint['yaws'][ids[aid]][k]).reshape(-1)[0])
            others=np.concatenate([dynamic,np.array([o['center'][:2] for o in static]).reshape(-1,2)])
            if len(others):others=others.copy();others[:,0]+=shift
            safe&=~overlap(centers,heading,ego[5:7],others,oy_current,sizes)
            jerk=(u-previous)/self.dt;cost+=((u[:,0]/4)**2+(u[:,1]/2.5)**2+.05*np.sum((jerk/5)**2,axis=1))/steps;previous=u
        progress=min(max(arc[-1]-initial_s,0),max(initial_v,0)*self.horizon*self.progress_ratio)
        road&=s-initial_s>=progress-1e-6;safe&=road
        return road,safe,cost

    def evaluate(self,snapshot):
        try:
            road,safe,cost=self._evaluate(snapshot)
            base=int(road.sum());count=int(safe.sum());out=dict(version=VERSION,reference_library_size=len(self.library),baseline_task_controls=base,safe_task_controls=count,measure='uniform_finite_two_phase_control_sequences',collision_outcome_used=False,off_route_outcome_used=False,scope='conditional_source_snapshot_reference_challenge_not_global_policy_independent_truth')
            if not base:return dict(out,valid=False,score=None,reason='reference_task_infeasible')
            compression=1-count/base
            if not count:return dict(out,valid=False,score=None,control_space_loss=compression,min_effort=None,reason='reference_library_exhausted_not_proven_physical_unsolvable')
            efforts=float(cost[safe].min());weights=[.4,.15,.15,.15,.15];variants=[(0,0,1),(.5,0,1),(-.5,0,1),(0,.2,1),(0,0,.8)];survival=[]
            for shift,delay,scale in variants:
                br,sa,_=self._evaluate(snapshot,shift,delay,scale)
                if not br.any():return dict(out,valid=False,score=None,reason='perturbed_reference_task_infeasible')
                survival.append(bool(sa.any()))
            fragility=1-sum(w*int(s) for w,s in zip(weights,survival));score=.5*compression+.3*efforts/(1+efforts)+.2*fragility
            return dict(out,valid=True,score=float(score),score_100=100*float(score),control_space_loss=compression,min_effort=efforts,fragility=float(fragility),weights=[.5,.3,.2],perturbation_weights=weights,reason=None)
        except (ValueError,KeyError,IndexError,TypeError) as error:return dict(version=VERSION,valid=False,score=None,reason='invalid_reference_input:'+type(error).__name__)


def apply_result(result,challenge):
    out=dict(result);score=challenge.get('score') if challenge.get('valid') else None
    out.update(historical_danger=result.get('scenario_danger_score'),scene_challenge=challenge,scene_challenge_score=score,outcome_severity_score=outcome_score(result),dual_track_protocol=VERSION,autonomous_driving_ability_score_legacy=result.get('autonomous_driving_ability_score'))
    out['autonomous_driving_ability_score']=ability_score(score,result.get('completed'),result.get('collision'),result.get('off_route'),result.get('progress')) if result.get('executed_steps',0)>0 and not result.get('runtime_failure') and not result.get('generation_failure') else None
    return out
