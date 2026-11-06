"""独立开发分支：同一扩散批次的连续可行候选回退，不改变离散计划或门槛。"""
import copy
from dataclasses import replace
import math
import time
from types import SimpleNamespace as NS
import numpy as np


def validate_candidate(env, candidate, project, states, dt, limits, settings, exempt):
    """沿用投影和公共安全检查；在提交前另核对原route_grid正面积条件。"""
    from policies.ego_profile import JointTrajectorySafetyGate
    from policies.joint_safety import NoSafeJointCandidate, validate_execution_prefix
    from policies.risk_metrics import RouteReachability
    corrected=project(candidate,states,dt,limits,settings,exempt)
    replay=NS(data_dict=env.data_dict,agent_active=env.agent_active,ego_state=env.ego_state,
        scenario_dict=env.scenario_dict,pending_joint_trajectory=corrected,dt=env.dt,
        previously_attacked_agent_ids=getattr(env,'previously_attacked_agent_ids',()),
        get_static_obstacles=env.get_static_obstacles)
    JointTrajectorySafetyGate(env.cfg.sim).validate_joint(replay)
    validate_execution_prefix(states,corrected,env.get_scenario_frame().static_obstacles_global)
    reach=RouteReachability(env.cfg.sim.evaluation.reachability)
    baseline=reach.compute(replay,False,env.get_static_obstacles())
    dangerous=reach.compute(replay,True,env.get_static_obstacles())
    if not (math.isfinite(baseline['area_m2']) and baseline['area_m2']>0
            and math.isfinite(dangerous['area_m2']) and dangerous['area_m2']>0 and dangerous['solvable']):
        raise NoSafeJointCandidate('riskweaver_zero_or_invalid_theoretical_reachable_area')
    return corrected,dict(baseline_area_m2=baseline['area_m2'],dangerous_area_m2=dangerous['area_m2'])


def recover_joint(env, native, candidates, project, states, dt, limits, settings, exempt=(),
                  enabled=False, max_candidates=7, max_seconds=20., validator=None, clock=time.monotonic):
    """先检原选择，失败才检原批次其他整组样本；绝不拼接各车或使用合成样本0。"""
    if not enabled:
        return project(native,states,dt,limits,settings,exempt)
    if not 1<=max_candidates<=31 or not math.isfinite(max_seconds) or max_seconds<=0:
        raise ValueError('invalid joint recovery budget')
    from policies.joint_safety import NoSafeJointCandidate
    validator=validator or validate_candidate
    started=clock();attempts=[];original_error=None
    selected=native.metadata.get('joint_sample_index')
    # 固定样本编号顺序，不依据真实未来或事后碰撞结果选择。
    options=[native]+[c for c in candidates if c.metadata.get('joint_sample_index') not in (0,selected)][:max_candidates]
    for index,candidate in enumerate(options):
        if index and clock()-started>=max_seconds:break
        if candidate.source_step!=native.source_step or not np.array_equal(candidate.agent_ids,native.agent_ids):
            attempts.append(dict(sample_index=candidate.metadata.get('joint_sample_index'),reason='stale_or_incomplete_candidate'))
            continue
        try:
            corrected,metrics=validator(env,candidate,project,states,dt,limits,settings,exempt)
            elapsed=clock()-started
            if index and elapsed>max_seconds:
                attempts.append(dict(sample_index=candidate.metadata.get('joint_sample_index'),reason='recovery_budget_exhausted'))
                break
            metadata=dict(corrected.metadata,joint_candidate_recovery=dict(
                enabled=True,recovered=index>0,original_sample_index=selected,
                selected_sample_index=candidate.metadata.get('joint_sample_index'),attempts=attempts,
                elapsed_seconds=elapsed,max_seconds=max_seconds,max_candidates=max_candidates,
                additional_diffusion_samples=0,additional_LLM_calls=0,**metrics))
            return replace(corrected,metadata=metadata)
        except NoSafeJointCandidate as error:
            if index==0:original_error=error
            attempts.append(dict(sample_index=candidate.metadata.get('joint_sample_index'),reason=str(error)))
    # 失败详情可被运行器捕获；保留最初拒绝类型，不把失败伪装成no_attack。
    env.last_joint_recovery_failure=dict(attempts=attempts,elapsed_seconds=clock()-started,
        reason='no_feasible_candidate_in_original_batch_or_budget',original_rejection=str(original_error))
    if original_error is not None:raise original_error
    raise NoSafeJointCandidate('joint_recovery_no_valid_native_candidate')


def install_recovery(max_candidates=7,max_seconds=20.):
    """显式调用才安装，仅作用于当前独立进程；返回恢复函数，不改仓库默认实现。"""
    from policies.diffusion_model_wrapper import DiffusionModelWrapper
    import policies.trajectory_projection as projection
    from simulator import Simulator
    original_sample=DiffusionModelWrapper._sample_policy
    original_predict=DiffusionModelWrapper.predict
    original_prepare=Simulator.prepare_background_traffic
    original_project=projection.project_joint
    current={}

    def sample(wrapper,*args,**kwargs):
        action,info=original_sample(wrapper,*args,**kwargs)
        raw=info.get('action_samples') if isinstance(info,dict) else None
        if isinstance(raw,dict) and raw.get('positions') is not None and raw.get('yaws') is not None:
            wrapper._recovery_samples=(raw['positions'].detach().cpu().numpy(),raw['yaws'].detach().cpu().numpy())
        return action,info

    def predict(wrapper,frame,attack_intent=None):
        wrapper._recovery_samples=None;wrapper._recovery_candidates=()
        native=original_predict(wrapper,frame,attack_intent=attack_intent)
        raw=wrapper._recovery_samples;wrapper._recovery_samples=None
        # 多GPU分片或远车独立生成需要另外验证，不把局部批次误当完整联合批次。
        if raw is None or len(wrapper.policy_replicas)>1:return native
        controlled=wrapper._select_diffusion_agent_ids(frame,attack_intent)
        batch=wrapper.adapter.build(frame,controlled_agent_ids=controlled)
        if not np.array_equal(batch.row_to_agent_id,native.agent_ids):return native
        positions,yaws=raw;decoded=[]
        for index in range(1,positions.shape[1]):
            joint=wrapper.adapter.decode(frame,batch,positions[:,index],yaws[:,index])
            metadata=copy.deepcopy(native.metadata)
            metadata.update(joint_sample_index=index,recovery_original_sample_index=native.metadata.get('joint_sample_index'))
            # 原选择的锚点误差不能复制给新样本；另算目标的真实全局误差。
            metadata.pop('anchor_satisfaction',None)
            if attack_intent is not None:
                trajectory=joint.trajectory_for(attack_intent['target_id'])
                anchors=np.asarray(attack_intent['anchors'],float)
                interval=float(wrapper.anchor_interval_seconds)
                if trajectory is not None and len(anchors)>1:
                    target_times=np.arange(1,len(anchors))*interval
                    indices=np.rint(target_times/float(frame.dt)).astype(int)-1
                    valid=(indices>=0)&(indices<len(trajectory))
                    errors=np.linalg.norm(trajectory[indices[valid],:2]-anchors[1:][valid,:2],axis=1)
                    metadata['recovery_anchor_diagnostics']=dict(anchor_times_s=target_times[valid].tolist(),errors_m=errors.tolist())
            decoded.append(replace(joint,metadata=metadata))
        wrapper._recovery_candidates=tuple(decoded)
        return native

    def prepare(env,*args,**kwargs):
        result=original_prepare(env,*args,**kwargs);current['env']=env
        return result

    def project(joint,states,dt,limits,settings,exempt_agent_ids=()):
        env=current.get('env')
        if env is None or env.current_step!=joint.source_step:
            return original_project(joint,states,dt,limits,settings,exempt_agent_ids)
        candidates=getattr(env.diffusion_controller,'_recovery_candidates',())
        candidates=[c for c in candidates if c.source_step==joint.source_step]
        corrected=recover_joint(env,joint,candidates,original_project,states,dt,limits,settings,
                                exempt_agent_ids,enabled=True,max_candidates=max_candidates,max_seconds=max_seconds)
        chosen=corrected.metadata.get('joint_candidate_recovery',{}).get('selected_sample_index')
        if chosen!=joint.metadata.get('joint_sample_index'):
            # 下一帧缓存必须沿所选同一联合样本继续，不能悄悄回到此前已拒绝的样本。
            match=next((c for c in candidates if c.metadata.get('joint_sample_index')==chosen),None)
            if match is not None:env._cached_joint_trajectory=match
        return corrected

    DiffusionModelWrapper._sample_policy=sample;DiffusionModelWrapper.predict=predict
    Simulator.prepare_background_traffic=prepare;projection.project_joint=project

    def restore():
        DiffusionModelWrapper._sample_policy=original_sample;DiffusionModelWrapper.predict=original_predict
        Simulator.prepare_background_traffic=original_prepare;projection.project_joint=original_project
    return restore
