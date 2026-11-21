import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from natural_worker import without_frame_cap
from campaign import structured_keys

def loop(count,args):
    while count < args.steps:
        count+=1
        if count==405:return count
    return count

def terminal(self,collided,completed):
    return collided or completed or self.t == self.cfg.sim.steps

def test_loop_reaches_past400():
    from types import SimpleNamespace
    patched,_=without_frame_cap(loop,'count < args.steps')
    assert loop(399,SimpleNamespace(steps=400))==400
    assert patched(399,SimpleNamespace(steps=400))==405

def test_only_time_cap_removed():
    from types import SimpleNamespace
    patched,_=without_frame_cap(terminal,'self.t == self.cfg.sim.steps')
    env=SimpleNamespace(t=400,cfg=SimpleNamespace(sim=SimpleNamespace(steps=400)))
    assert terminal(env,False,False)
    assert not patched(env,False,False)
    assert patched(env,True,False) and patched(env,False,True)

def test_unknown_patch_fails_closed():
    import pytest
    with pytest.raises(AssertionError):without_frame_cap(loop,'n < args.steps')

def test_rag_rule_mention_is_not_retrieval():
    import json
    off=[dict(content='Rules mention context.retrieved_attack_evidence.\n'+json.dumps(dict(context=dict(skill='context.retrieved_attack_evidence may occur'))))]
    keys,n=structured_keys(off);assert n==1 and 'retrieved_attack_evidence' not in keys
    on=[dict(content='rules\n'+json.dumps(dict(context=dict(retrieved_attack_evidence=[dict(case_id='real')]))))]
    assert 'retrieved_attack_evidence' in structured_keys(on)[0]
