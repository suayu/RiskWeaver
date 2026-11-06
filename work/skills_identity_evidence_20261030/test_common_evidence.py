"""哈希引用不是代填，比较事实不改变候选；真实SDK链替身继续严格拒绝错身份。"""
from pathlib import Path
import copy,importlib.util,json,sys
import pytest
import numpy as np
from common_evidence import add_facts,comparison_facts,render_prompt

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent


def sample(scene=74):
    import importlib.util
    spec=importlib.util.spec_from_file_location('base_event_prepare',ROOT/'work/skills_event_grid_20261030/prepare_planning.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return next(s for s in module.samples() if s['scene']==scene)


def test_correct_negative_direction_and_extreme_ids_do_not_reorder_catalog():
    context=sample()['contexts']['full_grid'];original=copy.deepcopy(context);new=add_facts(context)
    assert context==original and new['candidates']==context['candidates']
    facts=new.pop('comparison_facts');assert new==context
    r=facts['per_candidate']['event_0_t16']
    assert r['natural_min_minus_candidate_min_m']<0 and r['proxy_direction']=='farther_than_natural_CV'
    assert 'a0_8' in facts['smallest_active_gap']['candidate_ids']
    assert 'event_0_t16' not in facts['smallest_active_gap']['candidate_ids']


def test_two_difference_quantities_are_not_conflated():
    source=sample(140);facts=comparison_facts(source['contexts']['full_grid'])['per_candidate']['rolling_14']
    assert facts['natural_min_minus_candidate_min_m']==pytest.approx(.49029223323069)
    actor=np.asarray(source['state']['agents'][0]['state']);ego=np.asarray(source['state']['ego_state'])
    t=source['contexts']['full_grid']['incremental_cards']['rolling_14']['candidate_min_time_s']
    natural_at_t=float(np.linalg.norm(actor[:2]+t*actor[2:4]-ego[:2]-t*ego[2:4]))
    assert facts['natural_gap_at_candidate_min_time_minus_candidate_min_m']==pytest.approx(natural_at_t-facts['active_window_min_center_gap_m'])
    assert facts['natural_gap_at_candidate_min_time_minus_candidate_min_m']!=pytest.approx(facts['natural_min_minus_candidate_min_m'])


def test_null_margins_are_not_converted_to_zero_or_finite_best():
    facts=comparison_facts(sample()['contexts']['full_grid'])
    assert facts['largest_finite_background_margin']['value'] is None
    assert facts['largest_finite_background_margin']['candidate_ids']==[]


def test_full_identity_preserved_and_repeated_last_for_original_and_repair():
    values=dict(request_id='r'*32,state_hash='a'*64,catalog_hash='b'*64)
    context=add_facts(sample()['contexts']['full_grid'])
    payload=dict(**values,context=context,repair_feedback=dict(reason='skills_identity_mismatch',previous_response=dict(state_hash='wrong')))
    original=copy.deepcopy(payload);prompt=render_prompt('prefix\n'+json.dumps(payload))
    parsed=json.loads(prompt.rsplit('\n',1)[1])
    assert list(parsed)[-1]=='required_identity_copy_exactly' and parsed['required_identity_copy_exactly']==values
    assert parsed['repair_feedback']['expected_identity']==values and payload==original
    assert parsed['repair_feedback']['previous_response']['state_hash']=='wrong'
    assert {k:parsed[k] for k in values}==values


def runner_and_factory(bad=False):
    spec=importlib.util.spec_from_file_location('identity_frozen_runner',HERE/'frozen_planning.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    spec=importlib.util.spec_from_file_location('base_sdk_test_helper',ROOT/'work/skills_event_grid_20261030/test_frozen_planning.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return runner,helper.factory(bad),helper.config()


@pytest.mark.parametrize('mode',['temporal_original','full_grid'])
def test_both_real_prompt_chains_get_common_layer_and_compile(tmp_path,mode):
    runner,factory,cfg=runner_and_factory();r=runner.run_trial(sample(),mode,['deepseek-v4.1-flash-1017'],cfg,tmp_path/mode,planner_factory=factory)
    assert r['accepted_plan'] and r['budget']['transport_calls']==1
    assert r['planner_trace']['common_identity_evidence']['response_identity_not_rewritten']
    journal=[json.loads(s) for s in (tmp_path/mode/'api_calls.jsonl').read_text().splitlines()]
    payload=json.loads(journal[0]['messages'][-1]['content'].rsplit('\n',1)[1])
    assert 'comparison_facts' in payload['context']
    assert payload['required_identity_copy_exactly']['state_hash']==payload['state_hash']


def test_wrong_response_identity_still_rejected_after_common_hints(tmp_path):
    runner,factory,cfg=runner_and_factory(bad=True);r=runner.run_trial(sample(),'full_grid',['deepseek-v4.1-flash-1017'],cfg,tmp_path/'bad',planner_factory=factory)
    assert not r['accepted_plan'] and r['budget']['transport_calls']==2
    assert r['planner_trace']['validation_reason']=='skills_identity_mismatch'
