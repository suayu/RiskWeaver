"""候选级研究修正：单背景车也可按锚点完成度排序，默认不安装。"""
import numpy as np
import torch
from policies.joint_safety import NoSafeJointCandidate,safe_joint_candidates


def select_single_actor(wrapper,original,policy_net,model_batch,action,info,anchor_tensors=None):
    """只处理单背景车的已有样本，其他路径直接调用原函数。"""
    samples=info.get('action_samples') if isinstance(info,dict) else None
    positions=samples.get('positions') if isinstance(samples,dict) else None
    yaws=samples.get('yaws') if isinstance(samples,dict) else None
    if (not getattr(wrapper,'guidance_enabled',False) or not getattr(wrapper,'anchor_drive_enabled',False)
        or anchor_tensors is None or positions is None or yaws is None or positions.ndim!=4
        or positions.shape[0]!=1 or positions.shape[1]<2
        or bool(model_batch.get('full_adversarial_method',False))):
        return original(wrapper,policy_net,model_batch,action,info,anchor_tensors)
    mask=anchor_tensors['llm_anchor_mask'].to(positions.device)
    if not bool(mask.any()):return original(wrapper,policy_net,model_batch,action,info,anchor_tensors)
    # 原动作必须在候选集中，避免在缺少原基准动作时盲目替换。
    represented=torch.isclose(positions,action.positions[:,None],rtol=1e-5,atol=1e-5).reshape(1,positions.shape[1],-1).all(-1)[0]
    if not bool(represented.any()):return original(wrapper,policy_net,model_batch,action,info,anchor_tensors)
    _,count,horizon,_=positions.shape
    wrapper._last_anchor_audit=None
    with torch.no_grad():
        state=torch.cat([positions,torch.zeros_like(positions[...,:1]),yaws[...,:1]],dim=-1).reshape(count,horizon,4)
        data=policy_net._prepare_guidance_data(model_batch)
        native=policy_net.Loss_Calculater.calculate_loss(torch.zeros_like(state[...,:2]),state,data).reshape(1,count,horizon).sum((0,2))
        safe=safe_joint_candidates(positions.cpu().numpy(),yaws.cpu().numpy(),
            data['world_from_agent'].cpu().numpy(),data['ego_extents'].cpu().numpy())
        targets=anchor_tensors['llm_anchor_positions_local'].to(positions.device)
        weights=anchor_tensors['llm_anchor_weights'].to(positions.device)
        errors=torch.linalg.norm(positions-targets[:,None],dim=-1)
        weighted_mask=mask[:,None].to(errors.dtype)*weights[:,None]
        anchor=(errors*weighted_mask).sum((0,2))/weighted_mask.sum((0,2)).clamp_min(1e-6)
        raw=native+wrapper.anchor_candidate_score_weight*anchor
        # 最后一次屏蔽同时覆盖安全拒绝和NaN，避免inf+NaN使argmin选中无效候选。
        valid=torch.as_tensor(safe,device=raw.device)&torch.isfinite(raw)
        scores=torch.where(valid,raw,torch.full_like(raw,torch.inf))
        if not bool(valid.any()):raise NoSafeJointCandidate('single_actor_anchor_no_safe_finite_candidate')
        selected=int(torch.argmin(scores))
        chosen=errors[:,selected][mask]
        wrapper._last_anchor_audit=dict(mean_error_m=float(chosen.mean()),max_error_m=float(chosen.max()),
            candidate_score=float(anchor[selected]),selection_scope='single_actor_anchor_research',
            selected_index=selected,safe_finite_candidate_count=int(valid.sum()))
    return type(action)(positions=positions[:,selected],yaws=yaws[:,selected]),selected
