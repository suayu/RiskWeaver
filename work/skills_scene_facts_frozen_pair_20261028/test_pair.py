"""冻结状态对照必须保持目录与状态；类型化层不能绕过原身份和语义校验。"""
from pathlib import Path
import copy
import json

import pytest
from frozen_pair import make_view,validator_for,ROOT


def samples():
    return json.loads((ROOT/'work/skills_scene_facts_frozen4_protocol_20261028_r1/inputs.json').read_text())


def test_all_four_paired_views_differ_only_by_declared_fact_layer():
    for sample in samples():
        off_prompt,off=make_view(sample,False,'same-fixture-id')
        on_prompt,on=make_view(sample,True,'same-fixture-id')
        assert off['state_hash']==on['state_hash']
        assert off['context']['candidates']==on['context']['candidates']
        normalized=copy.deepcopy(on)
        del normalized['context']['scene_fact_cards'];del normalized['context']['scene_fact_validation_scope']
        normalized['catalog_hash']=off['catalog_hash']
        assert normalized==off
        assert 'scene_fact_id, source_step' not in off_prompt
        assert 'scene_fact_id, source_step' in on_prompt


def test_input_state_mismatch_does_not_keep_historical_hash():
    sample=copy.deepcopy(samples()[0]);_,old=make_view(sample,True,'r')
    sample['state']['ego_state'][0]+=1
    _,new=make_view(sample,True,'r')
    assert old['state_hash']!=new['state_hash']


def test_no_historical_response_in_new_input_and_no_prior_repair():
    for sample in samples():
        assert 'response' not in sample and sample['payload']['repair_feedback'] is None
        assert sample['source_step']==sample['state']['current_step']


def test_typed_claims_never_replace_original_identity_validation():
    sample=samples()[0];_,payload=make_view(sample,True,'r');engine=validator_for(payload,True)
    c=payload['context']['candidates'][0];cid=c['candidate_id'];card=payload['context']['scene_fact_cards'][cid]
    row=dict(request_id='r',state_hash=payload['state_hash'],catalog_hash=payload['catalog_hash'],ranking=[cid],
        scene_analysis='fixture',candidate_reasons={cid:'fixture'},no_attack_reason='',grounding={cid:dict(target_id=c['target_id'],target_role='background_vehicle',
        active_horizon_seconds=1.,evidence_ids=[cid+':actor',cid+':validity',cid+':active',cid+':contrast'],interaction_basis='exploratory_probe',
        contrast_direction=payload['context']['incremental_cards'][cid]['proxy_direction'],scene_fact_id=card['fact_id'],source_step=card['source_step'],
        longitudinal_relation=card['longitudinal_relation'],lateral_relation=card['lateral_relation'],heading_relation=card['heading_relation'])})
    catalog={c['candidate_id']:c for c in payload['context']['candidates']}
    engine.validate_response(row,'r',payload['state_hash'],payload['catalog_hash'],catalog,False)
    row['request_id']='old'
    with pytest.raises(ValueError):engine.validate_response(row,'r',payload['state_hash'],payload['catalog_hash'],catalog,False)
