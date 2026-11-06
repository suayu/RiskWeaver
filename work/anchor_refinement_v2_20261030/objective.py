"""独立进程中的可回退锚点目标；不改预训练网络、求解器或联合安全门槛。"""
from contextlib import contextmanager
import numpy as np
import torch

MODES=('legacy_dense_huber','dense_squared','sparse_squared','dense_squared_variance_floor')


def critical_mask(intent,step,dt,horizon,rows,target_row,interval=1.):
    """t=0为当前位置，不作为预测首帧；偏移严格使用(t-elapsed)/dt-1。"""
    mask=np.zeros((rows,horizon),bool)
    if intent is None or target_row is None:return mask
    times=np.asarray(intent.get('anchor_times_s',np.arange(len(intent['anchors']))*interval),float)
    if (times.shape!=(len(intent['anchors']),) or not np.isfinite(times).all()
            or np.any(np.diff(times)<=0) or dt<=0 or step<intent['source_step']):raise ValueError('invalid_anchor_timebase')
    elapsed=(step-intent['source_step'])*dt
    for time in times:
        offset=(time-elapsed)/dt
        if offset<=1e-7 or offset>horizon+1e-7:continue
        if abs(offset-round(offset))>1e-5:raise ValueError('prototype_requires_grid_aligned_anchor_times')
        mask[target_row,int(round(offset))-1]=True
    return mask


def squared_loss(calculator,action,state,data,sparse):
    """平方项避免Huber远距离饱和；输出形状与原损失完全相同，不改变外部权重。"""
    from tbsim.utils.trajdata_utils import enlarge_batch_samples
    if calculator.anchor_positions is None or calculator.anchor_mask is None:return state[...,0]*0.
    b=int(data['batch_size']);n=calculator.num_samples
    if calculator.anchor_positions.shape[0]!=b:raise ValueError('anchor_batch_mismatch')
    target=enlarge_batch_samples(calculator.anchor_positions.to(state),b,n)
    mask=calculator.anchor_mask
    if sparse:
        if not hasattr(calculator,'event_mask'):raise ValueError('missing_sparse_anchor_event_mask')
        mask=mask & calculator.event_mask.to(mask.device)
    mask=enlarge_batch_samples(mask.to(device=state.device,dtype=torch.bool),b,n)
    h=min(state.shape[1],target.shape[1]);loss=.5*(state[:,:h,:2]-target[:,:h]).square().sum(-1)
    loss=loss*mask[:,:h].to(loss.dtype)
    if calculator.anchor_weights is not None:
        weights=enlarge_batch_samples(calculator.anchor_weights.to(state),b,n);loss=loss*weights[:,:h]
    if calculator.loss_timesteps is not None:loss=loss*(torch.arange(h,device=state.device)<calculator.loss_timesteps)
    if h<state.shape[1]:loss=torch.nn.functional.pad(loss,(0,state.shape[1]-h))
    return loss*calculator.loss_scale*getattr(calculator,'gradient_variance_gain',1.)


@contextmanager
def install_anchor_objective(mode):
    """仅显式实验进程装配；退出或异常时恢复原类方法，默认路径零修改。"""
    if mode not in MODES:raise ValueError('unknown_anchor_objective')
    if mode==MODES[0]:yield;return
    from policies.llm_anchor_guidance import register_llm_anchor_guidance
    from policies.safe_sim_adapter import SafeSimBatchAdapter
    from policies.diffusion_model_wrapper import DiffusionModelWrapper
    from tbsim.models.diffusion import DiffusionTraj
    calculator=register_llm_anchor_guidance()
    old_loss=calculator.calculate_loss;old_build=SafeSimBatchAdapter.build_anchor_guidance;old_configure=DiffusionModelWrapper._configure_anchor_loss
    old_step=DiffusionTraj.n_step_guided_p_sample
    def loss(self,action,state,data):return squared_loss(self,action,state,data,mode=='sparse_squared')
    def build(self,frame,safe_batch,attack_intent,prediction_horizon,anchor_interval_seconds,anchor_phase_weights=None):
        tensors,metadata=old_build(self,frame,safe_batch,attack_intent,prediction_horizon,anchor_interval_seconds,anchor_phase_weights)
        ids=np.flatnonzero(safe_batch.row_to_agent_id==attack_intent['target_id']) if attack_intent else []
        event=critical_mask(attack_intent,frame.step,frame.dt,prediction_horizon,len(safe_batch.row_to_agent_id),int(ids[0]) if len(ids) else None,anchor_interval_seconds)
        tensors['llm_anchor_event_mask']=torch.from_numpy(event)
        metadata=dict(metadata,refinement_objective=mode,critical_predicted_frame_indices=np.flatnonzero(event.any(0)).tolist())
        return tensors,metadata
    def configure(self,net,tensors,active,device):
        obj=old_configure(self,net,tensors,active,device)
        if obj is not None:
            obj.event_mask=tensors['llm_anchor_event_mask'].to(device) if active else torch.zeros_like(obj.anchor_mask) if obj.anchor_mask is not None else None
        return obj
    def guided_step(self,*args,**kwargs):
        guide=kwargs['guide'];obj=guide.loss_calculator_dict.get('llm_anchor')
        if obj is None:raise ValueError('missing_anchor_calculator_for_variance_compensation')
        if not kwargs.get('scale_grad_by_std'):raise ValueError('variance_compensation_requires_original_std_scaled_guidance')
        beta=kwargs['beta']
        if not torch.isfinite(beta).all() or bool((beta<=0).any()):raise ValueError('invalid_diffusion_variance')
        # 只补偿锚点项的末期衰减，不放大背景/道路/接触项；滤样时恢复原尺度。
        # floor=.01及最大20倍在三种子结果前固定，不能通过调参丢弃不利种子。
        prior=getattr(obj,'gradient_variance_gain',None)
        obj.gradient_variance_gain=torch.clamp(.01/beta,min=1.,max=20.)
        try:return old_step(self,*args,**kwargs)
        finally:
            if prior is None:del obj.gradient_variance_gain
            else:obj.gradient_variance_gain=prior
    calculator.calculate_loss=loss;SafeSimBatchAdapter.build_anchor_guidance=build;DiffusionModelWrapper._configure_anchor_loss=configure
    if mode=='dense_squared_variance_floor':DiffusionTraj.n_step_guided_p_sample=guided_step
    try:yield
    finally:
        calculator.calculate_loss=old_loss;SafeSimBatchAdapter.build_anchor_guidance=old_build;DiffusionModelWrapper._configure_anchor_loss=old_configure
        DiffusionTraj.n_step_guided_p_sample=old_step
