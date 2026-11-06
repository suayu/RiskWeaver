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


def test_variance_floor_only_inside_gradient_step_and_restored_for_filter(monkeypatch):
    from tbsim.models.diffusion import DiffusionTraj
    obj=NS();seen=[]
    def base(self,*args,**kwargs):seen.append(float(obj.gradient_variance_gain));return 'unchanged_step'
    monkeypatch.setattr(DiffusionTraj,'n_step_guided_p_sample',base)
    guide=NS(loss_calculator_dict={'llm_anchor':obj})
    with install_anchor_objective('dense_squared_variance_floor'):
        result=DiffusionTraj.n_step_guided_p_sample(None,guide=guide,beta=torch.tensor(.001),scale_grad_by_std=True)
        assert result=='unchanged_step' and seen==pytest.approx([10.]) and not hasattr(obj,'gradient_variance_gain')
    assert DiffusionTraj.n_step_guided_p_sample is base


def test_variance_gain_does_not_change_non_anchor_loss_or_allow_unbounded_boost(monkeypatch):
    from tbsim.models.diffusion import DiffusionTraj
    obj=NS();other=NS();seen=[]
    def base(self,*args,**kwargs):seen.append(float(obj.gradient_variance_gain));assert not hasattr(other,'gradient_variance_gain')
    monkeypatch.setattr(DiffusionTraj,'n_step_guided_p_sample',base)
    guide=NS(loss_calculator_dict={'llm_anchor':obj,'route':other})
    with install_anchor_objective('dense_squared_variance_floor'):
        DiffusionTraj.n_step_guided_p_sample(None,guide=guide,beta=torch.tensor(.00001),scale_grad_by_std=True)
    assert seen==[20.] and not hasattr(obj,'gradient_variance_gain')


def test_proximal_reconstructs_feasible_unicycle_anchors_with_finite_controls():
    from proximal import fit_actions
    from tbsim.dynamics.unicycle import Unicycle
    dyn=Unicycle();states=torch.tensor([[0.,0.,5.,0.]]).repeat(3,1)
    base=torch.zeros(3,32,2);desired=base.clone();desired[:,:,0]=.5;desired[:,:,1]=.03
    targets=dyn.forward_dynamics(states,desired,.1)[0][...,:2]
    fitted,audit=fit_actions(dyn,states,base,targets,[9,19,29],.1)
    assert torch.isfinite(fitted).all()
    assert max(audit['final_event_squared_error_m2'])<.01
    assert all(y<x for x,y in zip(audit['initial_event_squared_error_m2'],audit['final_event_squared_error_m2']))
    costs=np.asarray(audit['regularized_cost_by_iteration'])
    assert (np.diff(costs,axis=0)<=1e-5).all()


def test_proximal_preserves_non_target_genuine_joint_samples_and_shape():
    from proximal import refine_samples
    from tbsim.dynamics.unicycle import Unicycle
    dyn=Unicycle();states=torch.tensor([[0.,0.,5.,0.],[0.,0.,4.,0.]])
    actions=torch.zeros(2,3,32,2);c=calculator();desired=actions[1].clone();desired[...,0]=.4
    c.anchor_positions[1]=dyn.forward_dynamics(states[1:].expand(3,-1),desired,.1)[0][0,:,:2]
    guide=NS(loss_calculator_dict={'llm_anchor':c},filter=lambda a,s,d:a.reshape(2,3,32,2)[:,1:2])
    sampler=NS(net=NS(dyn=dyn,step_time=.1))
    result=refine_samples(sampler,{'sampled_trajectories':actions},states,guide,{'batch_size':2})
    out=result['sampled_trajectories'];assert out.shape==actions.shape
    assert torch.equal(out[0,1:],actions[0,1:]) and not torch.equal(out[1,1:],actions[1,1:])
    assert torch.equal(actions,torch.zeros_like(actions))


def test_proximal_nonfinite_input_fails_closed():
    from proximal import fit_actions
    from tbsim.dynamics.unicycle import Unicycle
    with pytest.raises(ValueError,match='nonfinite'):
        fit_actions(Unicycle(),torch.tensor([[0.,0.,5.,0.]]),torch.full((1,32,2),float('nan')),torch.zeros(1,32,2),[9],.1)


def test_proximal_sample_and_joint_selector_restored_on_error():
    from tbsim.models.diffusion import DiffusionTraj
    from policies.diffusion_model_wrapper import DiffusionModelWrapper
    before=DiffusionTraj.sample,DiffusionModelWrapper._select_joint_guided_action
    with pytest.raises(RuntimeError):
        with install_anchor_objective('dense_squared_proximal'):
            assert DiffusionTraj.sample is not before[0];raise RuntimeError('restore')
    assert before==(DiffusionTraj.sample,DiffusionModelWrapper._select_joint_guided_action)
