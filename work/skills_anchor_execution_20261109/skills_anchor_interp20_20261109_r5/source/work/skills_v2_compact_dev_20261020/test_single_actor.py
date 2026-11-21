"""用真实原函数复现单背景车提前返回，并检验候选级修正的安全边界。"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from policies.diffusion_model_wrapper import DiffusionModelWrapper
from policies.joint_safety import NoSafeJointCandidate
import single_actor_anchor_selection as research


def fixture():
    wrapper=DiffusionModelWrapper.__new__(DiffusionModelWrapper)
    wrapper.guidance_enabled=True;wrapper.anchor_drive_enabled=True;wrapper.anchor_candidate_score_weight=10.
    positions=torch.zeros(1,3,2,2);positions[:,0,:,0]=10.;positions[:,1,:,0]=1.;positions[:,2,:,0]=5.
    yaws=torch.zeros(1,3,2,1)
    action=NS(positions=positions[:,0].clone(),yaws=yaws[:,0].clone())
    class Net:
        Loss_Calculater=NS(calculate_loss=lambda action,state,data:torch.zeros_like(state[...,0]))
        @staticmethod
        def _prepare_guidance_data(batch):
            return dict(batch_size=1,world_from_agent=torch.eye(3).repeat(3,1,1),ego_extents=torch.tensor([[4.,2.]]))
    anchors=dict(llm_anchor_positions_local=torch.zeros(1,2,2),llm_anchor_mask=torch.ones(1,2,dtype=torch.bool),llm_anchor_weights=torch.ones(1,2))
    return wrapper,Net(),action,dict(action_samples=dict(positions=positions,yaws=yaws)),anchors


def test_actual_original_skips_one_actor_reranking():
    wrapper,net,action,info,anchors=fixture()
    result,index=wrapper._select_joint_guided_action(net,{},action,info,anchors)
    assert result is action and index is None and wrapper._last_anchor_audit is None


def test_research_selects_closer_existing_candidate_without_changing_trajectory():
    wrapper,net,action,info,anchors=fixture()
    result,index=research.select_single_actor(wrapper,DiffusionModelWrapper._select_joint_guided_action,net,{},action,info,anchors)
    assert index==1 and wrapper._last_anchor_audit['mean_error_m']==1.
    torch.testing.assert_close(result.positions,info['action_samples']['positions'][:,1])


def test_safety_rejected_or_nonfinite_candidate_cannot_win(monkeypatch):
    wrapper,net,action,info,anchors=fixture()
    monkeypatch.setattr(research,'safe_joint_candidates',lambda *args:np.array([True,False,True]))
    result,index=research.select_single_actor(wrapper,DiffusionModelWrapper._select_joint_guided_action,net,{},action,info,anchors)
    assert index==2
    info['action_samples']['positions'][:,1]=float('nan')
    monkeypatch.setattr(research,'safe_joint_candidates',lambda *args:np.ones(3,dtype=bool))
    result,index=research.select_single_actor(wrapper,DiffusionModelWrapper._select_joint_guided_action,net,{},action,info,anchors)
    assert index==2


def test_all_unsafe_raises_instead_of_relaxing_threshold(monkeypatch):
    wrapper,net,action,info,anchors=fixture()
    monkeypatch.setattr(research,'safe_joint_candidates',lambda *args:np.zeros(3,dtype=bool))
    with pytest.raises(NoSafeJointCandidate):
        research.select_single_actor(wrapper,DiffusionModelWrapper._select_joint_guided_action,net,{},action,info,anchors)


def test_disabled_anchor_drive_preserves_original_result():
    wrapper,net,action,info,anchors=fixture();wrapper.anchor_drive_enabled=False
    result,index=research.select_single_actor(wrapper,DiffusionModelWrapper._select_joint_guided_action,net,{},action,info,anchors)
    assert result is action and index is None
