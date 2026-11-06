import copy,json
from legacy_attribution import recover


def example():
    p=dict(attack=True,attack_target_id=2,strategy='cut_in',reason='x',obstacle_plan=[])
    rows=[dict(kind='llm_output',step=0,validated_plan=p),dict(kind='plan_accepted',step=0,target_id=2,request_id=None),
          dict(kind='attack_execution',step=1,target_id=2,request_id=None,executed=True,diffusion_source_step=0)]
    calls=[dict(scenario_index=15,source_step=0,model_returned='m',request_id='r',content=json.dumps(p))]
    return rows,calls


def test_unique_and_original_unchanged():
    rows,calls=example();original=copy.deepcopy(rows);result,ledger=recover(rows,calls,15)
    assert result[-1]['request_id']=='r' and rows==original and len(ledger['bindings'])==1


def test_ambiguous_not_assigned():
    rows,calls=example();result,ledger=recover(rows,calls*2,15)
    assert result[-1]['request_id'] is None and ledger['unresolved']


def test_wrong_target_and_expired_not_assigned():
    rows,calls=example();rows[-1]['target_id']=3
    assert recover(rows,calls,15)[0][-1]['request_id'] is None
    rows[-1]['target_id']=2;rows[-1]['step']=11
    assert recover(rows,calls,15)[0][-1]['request_id'] is None
