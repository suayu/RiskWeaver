"""历史未来恢复的时效、身份、完整性、预算与钩子顺序；不请求模型。"""
from dataclasses import replace
from types import SimpleNamespace as NS
import copy
import numpy as np
import pytest
from policies.traffic_types import JointTrajectory
from policies.joint_safety import NoSafeJointCandidate
from temporal_joint_recovery import TemporalJointRecovery,install_temporal_recovery,_context


def joint(n=31,metadata=None):
    x=np.arange(1,n+1,dtype=float)*.1
    return JointTrajectory(source_step=0,agent_ids=np.array([0]),positions_global=np.stack([x,np.zeros_like(x)],axis=-1)[None],
        velocities_global=np.tile([1.,0.],(1,n,1)),yaws_global=np.zeros((1,n,1)),valid_mask=np.ones((1,n),bool),
        metadata=dict(joint_sample_index=3,anchor_guidance_active=False,adversarial_guidance_active=False,**(metadata or {})))


def env(now=1):
    e=NS(current_step=now,dt=.1,agent_active=np.array([True]),attack_intent=None,
        data_dict={'agent':[np.array([[.1,0.,1.,0.,0.,4.5,1.8,1.]])]},pending_joint_trajectory=None,
        cfg=NS(sim=NS(evaluation=NS(reachability=NS(horizon_seconds=3.)))))
    e.get_scenario_frame=lambda:NS(static_obstacles_global=np.empty((0,5)))
    return e


def good(env,candidate,project):return candidate,dict(validator='unit replacement; real recorded state tested separately')


def prepared(n=31):
    e=env();m=TemporalJointRecovery(validator=good);m.remember(e,joint(n),_context(e));return e,m


def test_same_whole_joint_is_trimmed_not_extrapolated_or_mixed():
    e,m=prepared();old=m.cache['joint'].positions_global.copy();r=m.recover(e)
    assert r.source_step==1 and r.positions_global.shape==(1,30,2)
    np.testing.assert_array_equal(r.positions_global,old[:,1:])
    assert r.metadata['temporal_joint_recovery']['additional_same_frame_diffusion_samples']==0
    assert r.metadata['temporal_joint_recovery']['no_extrapolation']
    np.testing.assert_array_equal(m.cache['joint'].positions_global,old)


@pytest.mark.parametrize('change,reason',[
 ('time','not_previous'),('active','active_set_changed'),('static','context_changed_static'),
 ('intent','context_changed_intent'),('horizon','incomplete_horizon'),('valid','incomplete_horizon'),
 ('consecutive','consecutive_limit'),('sample','synthetic_sample'),('stale_guidance','stale_attack_guidance')])
def test_stale_or_incomplete_cache_is_never_used(change,reason):
    e,m=prepared()
    if change=='time':e.current_step=2
    if change=='active':e.agent_active=np.array([True,True])
    if change=='static':e.get_scenario_frame=lambda:NS(static_obstacles_global=np.ones((1,5)))
    if change=='intent':e.attack_intent=dict(target_id=0,source_step=1,strategy='slow_down',anchors=[[0,0]],duration=10)
    if change=='horizon':m.cache['joint']=joint(30)
    if change=='valid':m.cache['joint'].valid_mask[0,30]=False
    if change=='consecutive':m.cache['joint'].metadata['temporal_joint_recovery']=dict(consecutive=1)
    if change=='sample':m.cache['joint'].metadata['joint_sample_index']=0
    if change=='stale_guidance':m.cache['joint'].metadata['anchor_guidance_active']=True
    with pytest.raises(NoSafeJointCandidate,match=reason):m.recover(e)


def test_expired_attack_does_not_gain_another_frame():
    e,m=prepared();e.attack_intent=dict(target_id=0,source_step=0,strategy='slow_down',anchors=[[0,0]],duration=1)
    m.cache['context']=_context(e)
    with pytest.raises(NoSafeJointCandidate,match='expired'):m.recover(e)


def test_changed_real_guidance_weights_invalidate_cached_future():
    # 不能在求解参数或引导权重变化后沿用旧未来，即使意图和参与者没有变化。
    class Weights:
        def __init__(self,value):self.value=value
        def detach(self):return self
        def cpu(self):return self
        def tolist(self):return self.value
    e,m=prepared()
    weights=Weights([1.5,1.,2.,4.])
    e.diffusion_controller=NS(policy=NS(nets={'policy':NS(guide_config=NS(params=dict(inner_lr=.2,inner_beta=.5,n_guide_steps=2)),
                                                        Loss_Calculater=NS(weights=weights))}))
    m.cache['context']=_context(e);weights.value=[1.5,1.,2.,8.]
    with pytest.raises(NoSafeJointCandidate,match='context_changed_guidance'):m.recover(e)


def test_validator_failure_or_acceptance_timeout_remains_rejected():
    e,m=prepared();m.validator=lambda *a:(_ for _ in ()).throw(NoSafeJointCandidate('background_margin'))
    with pytest.raises(NoSafeJointCandidate,match='background_margin'):m.recover(e)
    m.validator=good;clock=iter([0.,31.]);m.clock=lambda:next(clock)
    with pytest.raises(NoSafeJointCandidate,match='budget_exceeded'):m.recover(e)


def test_real_hook_uses_completed_transition_and_does_not_reuse_forever():
    class Fake:
        def __init__(self):self.__dict__.update(env(0).__dict__);self.calls=0
        def reset(self):self.current_step=0;self.calls=0
        def prepare_background_traffic(self):
            self.calls+=1
            if self.calls>1:raise NoSafeJointCandidate('new_batch_rejected')
            self.pending_joint_trajectory=joint();return self.pending_joint_trajectory
        def inject_joint_trajectory(self,j):self.pending_joint_trajectory=j
        def step(self,*args,**kwargs):
            self.current_step+=1;self.pending_joint_trajectory=None;return None,False,{}
    original=Fake.prepare_background_traffic
    restore=install_temporal_recovery(simulator_cls=Fake,validator=good)
    try:
        e=Fake();e.reset();e.prepare_background_traffic();e.step(None,None,None)
        tail=e.prepare_background_traffic()
        assert tail.source_step==1 and tail.metadata['temporal_joint_recovery']['consecutive']==1
        assert e._cached_joint_trajectory is None
        e.step(None,None,None)
        with pytest.raises(NoSafeJointCandidate,match='new_batch_rejected'):e.prepare_background_traffic()
    finally:restore()
    assert Fake.prepare_background_traffic is original
