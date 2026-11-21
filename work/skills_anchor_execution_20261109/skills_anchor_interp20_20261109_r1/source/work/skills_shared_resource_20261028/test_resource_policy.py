"""资源准入、夹具和门禁负例；不初始化GPU或发送API。"""
import copy
import math

import pytest
from resource_policy import eligible_shared,memory_fraction,fixture_response
from probe_acceptance import gate


def test_busy_gpu_with_enough_headroom_allowed_but_small_memory_not_allowed():
    rows='0,32768,18287,94\n1,32768,17297,95\n2,32768,24000,0'
    selected=eligible_shared(rows)
    assert [r['index'] for r in selected]==['0']
    assert eligible_shared(rows,reserved=['0'])==[]


def test_formal_slot_selection_prefers_most_free_and_never_uses_other_gpu():
    rows='0,32768,19000,99\n1,32768,22000,90\n2,32768,32000,0'
    assert [r['index'] for r in eligible_shared(rows)]==['1','0']


@pytest.mark.parametrize('cap',[0.,-1.,float('nan'),float('inf'),True])
def test_invalid_memory_admission_rejected(cap):
    with pytest.raises(ValueError):eligible_shared('0,32768,20000,0',cap_gib=cap)


def test_memory_cap_is_fraction_of_total_not_remaining_memory():
    assert memory_fraction(14.,32*2**30)==14/32
    with pytest.raises(ValueError):memory_fraction(32.,32*2**30)


def test_resource_fixture_has_only_current_catalog_and_never_counts_as_LLM():
    card=lambda difference,gap:dict(facts={'a:contrast':dict(min_gap_difference_m=difference,candidate_min_gap_m=gap,proxy_direction='closer_than_natural_CV')},active_horizon_seconds=1.)
    p=dict(request_id='r',state_hash='s',catalog_hash='c',context=dict(candidates=[dict(candidate_id='a',target_id=0)],priority_evidence={'a':card(1.,4.)}))
    old=copy.deepcopy(p);r=fixture_response(p)
    assert r['ranking']==['a'] and r['grounding']['a']['target_id']==0 and p==old
    assert 'not an LLM response' in r['scene_analysis']
    assert 'anchors' not in r and r['request_id']=='r'


def test_any_failed_or_missing_resource_check_blocks_admission():
    assert gate(dict(memory=True,guided=True))
    assert not gate(dict(memory=True,guided=False))
    assert not gate({})
    assert not gate(dict(memory=True,guided=None))


def test_probe_can_only_signal_its_own_verified_command(monkeypatch):
    from types import SimpleNamespace as NS
    import launch_probe
    events=[]
    class Proc:
        def __truediv__(self,other):return self
        def exists(self):return True
        def read_bytes(self):return b'python /home2/zhaoyx/scenario-dreamer/work/skills_shared_resource_20261028/run_capped.py --output-dir /experiment/probe'
    monkeypatch.setattr(launch_probe,'Path',lambda *args:Proc())
    child=NS(pid=123,poll=lambda:None,terminate=lambda:events.append('terminate'),wait=lambda timeout:0)
    launch_probe.stop_owned(child,'/experiment/probe')
    assert events==['terminate']
    with pytest.raises(RuntimeError):launch_probe.stop_owned(child,'/other_experiment')
    assert events==['terminate']


def test_sampling_observer_preserves_result_and_records_failure_without_retry(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    import numpy as np
    import torch
    from policies.diffusion_model_wrapper import DiffusionModelWrapper as Wrapper
    from run_capped import install_sampling_observer
    import json
    events=[];result=object()
    def sample(self,*args):events.append(args);return result
    monkeypatch.setattr(Wrapper,'_sample_policy',sample)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *args:128)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *args:256)
    wrapper=Wrapper.__new__(Wrapper);wrapper.scene_id='fixture';wrapper.num_samples=8
    path=tmp_path/'sample.jsonl';restore=install_sampling_observer(path)
    args=(object(),NS(type='cuda'),dict(image=np.zeros((2,1))),None,True)
    try:assert wrapper._sample_policy(*args) is result
    finally:restore()
    assert len(events)==1 and Wrapper._sample_policy is sample
    row=json.loads(path.read_text());assert row['success'] and row['num_samples']==8 and row['row_count']==2
