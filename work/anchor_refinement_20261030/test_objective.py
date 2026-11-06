"""真实损失/采样排列、时间偏移、零误差梯度与退出恢复；无模型/API调用。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from objective import critical_mask,squared_loss,install_anchor_objective


def intent():return dict(anchors=[[0,0]]*4,source_step=0,target_id=1)


def test_current_position_is_not_future_zero_index():
    assert np.flatnonzero(critical_mask(intent(),0,.1,32,1,0)[0]).tolist()==[9,19,29]


def test_elapsed_frame_and_explicit_time_horizon():
    assert np.flatnonzero(critical_mask(intent(),1,.1,32,1,0)[0]).tolist()==[8,18,28]
    assert not critical_mask(intent(),30,.1,32,1,0).any()


def test_non_grid_time_not_rounded_to_other_event():
    i=intent();i['anchor_times_s']=[0,.15,1,2]
    with pytest.raises(ValueError,match='grid'):critical_mask(i,0,.1,32,1,0)


def calculator():
    mask=torch.zeros(2,32,dtype=torch.bool);mask[1,:30]=True
    event=torch.zeros_like(mask);event[1,[9,19,29]]=True
    return NS(anchor_positions=torch.zeros(2,32,2),anchor_mask=mask,event_mask=event,
        anchor_weights=torch.ones(2,32),num_samples=3,loss_timesteps=None,loss_scale=1.)


def test_sparse_loss_only_targets_correct_agent_sample_and_anchor_times():
    c=calculator();state=torch.ones(6,32,4,requires_grad=True)
    loss=squared_loss(c,state[...,:2],state,dict(batch_size=2),True)
    assert loss.shape==(6,32) and loss[:3].sum()==0 and torch.count_nonzero(loss[3:])==9
    loss.sum().backward();assert torch.isfinite(state.grad).all() and state.grad[:3].abs().sum()==0


def test_zero_position_error_has_finite_zero_gradient():
    c=calculator();state=torch.zeros(6,32,4,requires_grad=True)
    squared_loss(c,state[...,:2],state,dict(batch_size=2),True).sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum()==0


def test_far_residual_gradient_grows_and_descent_reduces_objective():
    c=calculator();state=torch.full((6,32,4),10.,requires_grad=True)
    before=squared_loss(c,state[...,:2],state,dict(batch_size=2),True).sum()
    grad=torch.autograd.grad(before,state)[0]
    after=squared_loss(c,state[...,:2],state-.01*grad,dict(batch_size=2),True).sum()
    assert after<before and grad.max()==10


def test_mode_off_and_exception_restore_original_production_methods():
    from policies.llm_anchor_guidance import register_llm_anchor_guidance
    from policies.safe_sim_adapter import SafeSimBatchAdapter
    from policies.diffusion_model_wrapper import DiffusionModelWrapper
    c=register_llm_anchor_guidance();before=(c.calculate_loss,SafeSimBatchAdapter.build_anchor_guidance,DiffusionModelWrapper._configure_anchor_loss)
    with install_anchor_objective('legacy_dense_huber'):assert c.calculate_loss is before[0]
    with pytest.raises(RuntimeError):
        with install_anchor_objective('sparse_squared'):
            assert c.calculate_loss is not before[0];raise RuntimeError('test_restore')
    assert before==(c.calculate_loss,SafeSimBatchAdapter.build_anchor_guidance,DiffusionModelWrapper._configure_anchor_loss)
