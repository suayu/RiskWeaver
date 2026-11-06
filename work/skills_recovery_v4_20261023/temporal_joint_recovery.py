"""可选的整组历史未来恢复；每次重新检查，不补车、不外推、不延长攻击有效期。"""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import time
import numpy as np
from policies.joint_safety import NoSafeJointCandidate,validate_execution_prefix
from policies.attack_execution_gate import assess_reachability_execution


def _intent_key(intent):
    if intent is None:return None
    return (int(intent['target_id']),int(intent['source_step']),str(intent['strategy']),str(intent.get('request_id')),
            int(intent.get('duration',10)),np.asarray(intent['anchors'],float).tobytes())


def _context(env):
    frame=env.get_scenario_frame();policy=getattr(getattr(env,'diffusion_controller',None),'policy',None);guidance=None
    if policy is not None:
        net=policy.nets['policy'];params=net.guide_config.params
        guidance=tuple(params[k] for k in ['inner_lr','inner_beta','n_guide_steps'])
        if getattr(net,'Loss_Calculater',None) is not None:guidance+=tuple(net.Loss_Calculater.weights.detach().cpu().tolist())
    return dict(intent=_intent_key(getattr(env,'attack_intent',None)),static=frame.static_obstacles_global.tobytes(),guidance=guidance,dt=float(env.dt))


def validate_tail(env,candidate,project):
    """始终重新运行原动力学/背景/端点中点检查，并使用当前显式面积策略。"""
    from policies.ego_profile import JointTrajectorySafetyGate
    from policies.risk_metrics import RouteReachability
    from types import SimpleNamespace as NS
    cfg=env.cfg.sim.evaluation.reachability;timings={};started=time.monotonic()
    if str(getattr(cfg,'backend','route_grid'))!='route_grid':raise NoSafeJointCandidate('temporal_recovery_backend_not_supported')
    gate=JointTrajectorySafetyGate(env.cfg.sim)
    corrected=project(candidate,env.data_dict['agent'][-1],env.dt,gate.dynamic_limits,
        env.cfg.sim.traffic_model.dynamics_projection,getattr(env,'previously_attacked_agent_ids',()))
    timings['dynamics_projection_seconds']=time.monotonic()-started;started=time.monotonic()
    replay=NS(data_dict=env.data_dict,agent_active=env.agent_active,ego_state=env.ego_state,
        scenario_dict=env.scenario_dict,pending_joint_trajectory=corrected,dt=env.dt,
        previously_attacked_agent_ids=getattr(env,'previously_attacked_agent_ids',()),get_static_obstacles=env.get_static_obstacles)
    gate.validate_joint(replay)
    validate_execution_prefix(env.data_dict['agent'][-1],corrected,env.get_scenario_frame().static_obstacles_global)
    timings['background_and_execution_prefix_seconds']=time.monotonic()-started;started=time.monotonic()
    reach=RouteReachability(cfg);base=reach.compute(replay,False,env.get_static_obstacles())
    timings['CV_area_seconds']=time.monotonic()-started;started=time.monotonic()
    actual=reach.compute(replay,True,env.get_static_obstacles())
    timings['joint_area_seconds']=time.monotonic()-started
    event=dict(original_area_m2=base['area_m2'],dangerous_area_m2=actual['area_m2'],dangerous_solvable=actual['solvable'],
        difficulty=1-actual['area_m2']/base['area_m2'] if base['area_m2']>0 else float('nan'))
    verdict=assess_reachability_execution(event,str(getattr(cfg,'execution_gate_policy','legacy_relative')))
    if not verdict['allowed']:raise NoSafeJointCandidate('temporal_recovery_'+verdict['reason'])
    return corrected,dict(area_gate=verdict,validation_timings=timings)


class TemporalJointRecovery:
    def __init__(self,max_consecutive=1,max_seconds=30.,validator=None,project=None,clock=time.monotonic):
        if isinstance(max_consecutive,bool) or not isinstance(max_consecutive,int) or not 1<=max_consecutive<=3:raise ValueError('invalid consecutive budget')
        if not math.isfinite(max_seconds) or max_seconds<=0:raise ValueError('invalid validation budget')
        self.max_consecutive=max_consecutive;self.max_seconds=max_seconds;self.validator=validator or validate_tail
        self.project=project;self.clock=clock;self.cache=None

    def remember(self,env,joint,context):
        self.cache=None if joint is None else dict(joint=copy.deepcopy(joint),context=context,after_step=int(env.current_step))

    def recover(self,env):
        if self.cache is None:raise NoSafeJointCandidate('temporal_cache_missing')
        cached=self.cache['joint'];now=int(env.current_step);offset=now-int(cached.source_step)
        if offset!=1 or self.cache['after_step']!=now:raise NoSafeJointCandidate('temporal_cache_not_previous_transition')
        current=_context(env)
        for name in ['intent','static','guidance','dt']:
            if current[name]!=self.cache['context'][name]:raise NoSafeJointCandidate('temporal_cache_context_changed_'+name)
        active=np.flatnonzero(env.agent_active)
        if not len(active) or not np.array_equal(cached.agent_ids,active):raise NoSafeJointCandidate('temporal_cache_active_set_changed')
        sample=cached.metadata.get('joint_sample_index')
        if isinstance(sample,bool) or not isinstance(sample,(int,np.integer)) or sample<=0:
            raise NoSafeJointCandidate('temporal_cache_untracked_or_synthetic_sample')
        previous=cached.metadata.get('temporal_joint_recovery',{});consecutive=int(previous.get('consecutive',0))+1
        if consecutive>self.max_consecutive:raise NoSafeJointCandidate('temporal_cache_consecutive_limit')
        intent=getattr(env,'attack_intent',None)
        if intent is not None and not 0<=now-int(intent['source_step'])<int(intent.get('duration',10)):
            raise NoSafeJointCandidate('temporal_cache_attack_expired')
        if intent is None and (cached.metadata.get('anchor_guidance_active') or cached.metadata.get('adversarial_guidance_active')):
            raise NoSafeJointCandidate('temporal_cache_stale_attack_guidance')
        required=int(math.ceil(float(getattr(env.cfg.sim.evaluation.reachability,'horizon_seconds',3.))/env.dt))
        if cached.positions_global.shape[1]-offset<required or not np.all(cached.valid_mask[:,offset:offset+required]):
            raise NoSafeJointCandidate('temporal_cache_incomplete_horizon')
        metadata=copy.deepcopy(cached.metadata)
        for key in ['anchor_satisfaction','recovery_anchor_diagnostics','candidate_trajectories_global']:
            metadata.pop(key,None)
        metadata.update(candidate_trajectories_global=[],all_samples_available=False,replan_reused=True,replan_cache_offset=offset)
        if intent is not None:
            metadata['attack_age_frames']=now-int(intent['source_step'])
            if 'anchor_guidance_valid_steps' in metadata:metadata['anchor_guidance_valid_steps']=max(0,metadata['anchor_guidance_valid_steps']-offset)
        candidate=replace(cached,source_step=now,positions_global=cached.positions_global[:,offset:].copy(),
            velocities_global=cached.velocities_global[:,offset:].copy(),yaws_global=cached.yaws_global[:,offset:].copy(),
            valid_mask=cached.valid_mask[:,offset:].copy(),metadata=metadata)
        project=self.project
        if project is None:
            from policies.trajectory_projection import project_joint
            project=project_joint
        started=self.clock();corrected,metrics=self.validator(env,candidate,project);elapsed=self.clock()-started
        if elapsed>self.max_seconds:raise NoSafeJointCandidate(f'temporal_cache_acceptance_budget_exceeded:{elapsed:.3f}s>{self.max_seconds:.3f}s')
        metadata=dict(corrected.metadata,temporal_joint_recovery=dict(enabled=True,recovered=True,
            previous_source_step=int(cached.source_step),origin_source_step=previous.get('origin_source_step',int(cached.source_step)),
            consecutive=consecutive,max_consecutive=self.max_consecutive,remaining_steps=candidate.positions_global.shape[1],
            max_validation_seconds=self.max_seconds,validation_seconds=elapsed,additional_same_frame_diffusion_samples=0,
            additional_LLM_calls=0,no_extrapolation=True,all_active_agents_verified=True,**metrics))
        return replace(corrected,metadata=metadata)


def install_temporal_recovery(log_path=None,simulator_cls=None,max_consecutive=1,max_seconds=30.,validator=None):
    """仅在未来显式开启的进程中安装；保存实际完成转移的未来，下一次才可恢复。"""
    if simulator_cls is None:
        from simulator import Simulator
        simulator_cls=Simulator
    from policies.trajectory_projection import project_joint
    original_prepare=simulator_cls.prepare_background_traffic;original_step=simulator_cls.step;original_reset=simulator_cls.reset
    log_path=Path(log_path) if log_path is not None else None
    if log_path is not None and log_path.exists():raise FileExistsError(log_path)
    def state(env):
        if not hasattr(env,'_temporal_joint_manager'):env._temporal_joint_manager=TemporalJointRecovery(max_consecutive,max_seconds,validator,project_joint)
        return env._temporal_joint_manager
    def log(env,value):
        if log_path is not None:
            record=dict(scene_id=str(getattr(env,'current_scene_id','unknown')),step=int(env.current_step),**value)
            with log_path.open('a',encoding='utf8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    def reset(env,*args,**kwargs):
        env._temporal_joint_manager=TemporalJointRecovery(max_consecutive,max_seconds,validator,project_joint)
        return original_reset(env,*args,**kwargs)
    def step(env,*args,**kwargs):
        joint=getattr(env,'pending_joint_trajectory',None);context=_context(env) if joint is not None else None
        result=original_step(env,*args,**kwargs)
        state(env).remember(env,joint,context)
        return result
    def prepare(env,*args,**kwargs):
        try:return original_prepare(env,*args,**kwargs)
        except NoSafeJointCandidate as native_error:
            try:tail=state(env).recover(env)
            except NoSafeJointCandidate as recovery_error:
                log(env,dict(recovered=False,native_reason=str(native_error),recovery_reason=str(recovery_error)))
                raise native_error
            env.inject_joint_trajectory(tail)
            # 下一帧必须重新预测；不能把短历史尾段反复当成一个新的完整时域缓存。
            env._cached_joint_trajectory=None;env.latest_diffusion_candidate_trajectories=[]
            log(env,dict(recovered=True,native_reason=str(native_error),recovery=tail.metadata['temporal_joint_recovery']))
            return tail
    simulator_cls.prepare_background_traffic=prepare;simulator_cls.step=step;simulator_cls.reset=reset
    def restore():
        simulator_cls.prepare_background_traffic=original_prepare;simulator_cls.step=original_step;simulator_cls.reset=original_reset
    return restore
