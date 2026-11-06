"""回退的开关、原错误、联合样本身份和预算行为；不依赖模型权重。"""
from dataclasses import dataclass,field
from types import SimpleNamespace as NS
import numpy as np
import pytest
from policies.joint_safety import NoSafeJointCandidate
from joint_candidate_recovery import recover_joint


@dataclass
class Candidate:
    source_step:int=0
    agent_ids:object=field(default_factory=lambda:np.array([0,1]))
    metadata:dict=field(default_factory=dict)


def joint(index,step=0,ids=(0,1)):
    return Candidate(step,np.asarray(ids),dict(joint_sample_index=index))


def test_off_uses_only_old_projection():
    called=[];native=joint(2)
    def old(*args):called.append(args[0]);return native
    result=recover_joint(NS(),native,[joint(4)],old,None,.1,{},None)
    assert result is native and called==[native] and 'joint_candidate_recovery' not in result.metadata


def test_native_pass_does_not_search_alternatives():
    calls=[]
    def valid(env,c,*args):calls.append(c.metadata['joint_sample_index']);return c,{}
    result=recover_joint(NS(),joint(2),[joint(4)],None,None,.1,{},None,enabled=True,validator=valid)
    assert calls==[2] and result.metadata['joint_candidate_recovery']['recovered'] is False


def test_recovery_never_uses_synthetic_zero_or_duplicates():
    calls=[]
    def valid(env,c,*args):
        index=c.metadata['joint_sample_index'];calls.append(index)
        if index!=4:raise NoSafeJointCandidate('projection_excessive_deviation')
        return c,{'baseline_area_m2':2.,'dangerous_area_m2':1.}
    result=recover_joint(NS(),joint(2),[joint(0),joint(2),joint(3),joint(4)],None,None,.1,{},None,
                         enabled=True,validator=valid)
    assert calls==[2,3,4] and result.metadata['joint_sample_index']==4
    assert result.metadata['joint_candidate_recovery']['additional_diffusion_samples']==0


def test_failure_preserves_original_rejection():
    env=NS()
    def reject(env,c,*args):raise NoSafeJointCandidate('native_failure' if c.metadata['joint_sample_index']==2 else 'other_failure')
    with pytest.raises(NoSafeJointCandidate,match='native_failure'):
        recover_joint(env,joint(2),[joint(4)],None,None,.1,{},None,enabled=True,validator=reject)
    assert len(env.last_joint_recovery_failure['attempts'])==2


def test_stale_or_incomplete_candidates_not_validated():
    seen=[]
    def reject(env,c,*args):seen.append(c.metadata['joint_sample_index']);raise NoSafeJointCandidate('reject')
    with pytest.raises(NoSafeJointCandidate):
        recover_joint(NS(),joint(2),[joint(3,step=1),joint(4,ids=(0,))],None,None,.1,{},None,enabled=True,validator=reject)
    assert seen==[2]


def test_late_result_cannot_commit_after_budget():
    tick=[0.];env=NS()
    def valid(env,c,*args):
        tick[0]+=2.
        if c.metadata['joint_sample_index']==2:raise NoSafeJointCandidate('native_failure')
        return c,{}
    with pytest.raises(NoSafeJointCandidate,match='native_failure'):
        recover_joint(env,joint(2),[joint(4)],None,None,.1,{},None,enabled=True,max_seconds=3.,
                      validator=valid,clock=lambda:tick[0])
    assert env.last_joint_recovery_failure['attempts'][-1]['reason']=='recovery_budget_exhausted'


def test_unexpected_errors_are_not_hidden():
    def crash(*args):raise ValueError('invalid_geometry')
    with pytest.raises(ValueError,match='invalid_geometry'):
        recover_joint(NS(),joint(2),[],None,None,.1,{},None,enabled=True,validator=crash)
