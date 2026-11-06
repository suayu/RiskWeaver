"""可逆数据视图的语义保持、类型边界及请求生命周期；零API。"""
import copy,json
from types import SimpleNamespace as NS
import pytest
from factored_view import common_fields,encode_payload,expand_payload,render,install_factored_view,MARKER


def fixture():
    return dict(request_id='request',state_hash='s'*64,catalog_hash='c'*64,context=dict(candidates=[
        dict(candidate_id='a',target_id=0,duration=10,anchors=[[0.,0.],[1.,0.],[2.,0.],[3.,0.]],margin=None),
        dict(candidate_id='b',target_id=0,duration=10,anchors=[[0.,0.],[1.,1.],[2.,2.],[3.,3.]],margin=1.5)],
        priority_evidence={cid:dict(target_id=0,active_horizon_seconds=1.,facts={cid+':actor':dict(target_id=0,role='background_vehicle'),
        cid+':validity':dict(active_horizon_seconds=1.,anchor_horizon_seconds=3.),cid+':active':dict(executed=False,min_center_distance_m=value)}) for cid,value in [('a',-0.0),('b',5.25)]}),
        required_identity_copy_exactly=dict(request_id='request',state_hash='s'*64,catalog_hash='c'*64))


def test_exact_roundtrip_keeps_ids_order_null_negative_zero_and_original_unmodified():
    source=fixture();before=copy.deepcopy(source);encoded=encode_payload(source)
    assert json.dumps(expand_payload(encoded),sort_keys=True)==json.dumps(source,sort_keys=True)
    assert source==before and [r['candidate_id'] for r in encoded['context']['candidates']]==['a','b']
    assert expand_payload(encoded)['context']['candidates'][0]['margin'] is None


def test_boolean_integer_and_float_not_silently_merged():
    assert common_fields([{'v':True},{'v':1}])=={} and common_fields([{'v':1},{'v':1.}])=={}


def test_incomplete_shared_fields_stay_local_and_empty_objects_survive():
    data=fixture();data['context']['candidates'][0]['only_a']={}
    view=encode_payload(data)
    assert expand_payload(view)['context']['candidates'][0]['only_a']=={}


def test_identity_reference_stays_final_and_exact_not_regenerated():
    source=fixture();prompt='PREFIX\n'+json.dumps(source);result,receipt=render(prompt)
    payload=json.loads(result.rsplit('\n',1)[1])
    assert list(payload)[-1]=='required_identity_copy_exactly' and receipt['identity_fields_unchanged']
    assert payload['request_id']==source['request_id'] and receipt['payload_roundtrip_verified']


def test_reserved_marker_conflict_and_changed_cardinality_fail_explicitly():
    source=fixture();source['context'][MARKER]={}
    with pytest.raises(ValueError,match='reserved'):encode_payload(source)
    view=encode_payload(fixture());view['context']['candidates'].pop()
    with pytest.raises(ValueError,match='cardinality'):expand_payload(view)


def test_unsupported_original_free_planner_is_not_rewritten():
    assert render('original free prompt')[0]=='original free prompt'


def test_off_switch_preserves_bound_method_and_on_restores_after_exception():
    original=lambda planner:None;engine=NS(plan=original)
    assert install_factored_view(engine).plan is original
    def fails(planner):
        planner._request_attack_plan('PREFIX\n'+json.dumps(fixture()));raise RuntimeError('failure')
    engine.plan=fails;sent=[];request=lambda prompt,scene_image=None:sent.append(prompt)
    planner=NS(_request_attack_plan=request,last_trace={});install_factored_view(engine,True)
    with pytest.raises(RuntimeError):engine.plan(planner)
    assert planner._request_attack_plan is request and len(sent)==1 and planner.last_trace['factored_wire_view']['records'][0]['supported']
