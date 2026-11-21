"""覆盖旧验收器未能识别的同数量错场景、未执行计划与预算缺失。"""
from audit_enhanced import manifest_coverage,plan_execution_ledger,catalog_ledger,transport_ledger


def test_same_number_wrong_scenario_is_not_complete():
    manifest=dict(original_scenario_indices=[80],scenario_count=1,scenario_files=['/a/80.pkl'])
    assert not manifest_coverage(manifest,[dict(scenario_index=81,scenario_source='/a/81.pkl')])['verified']
    assert not manifest_coverage(manifest,[dict(scenario_index=80,scenario_source='/a/81.pkl')])['verified']
    assert manifest_coverage(manifest,[dict(scenario_index=80,scenario_source='/a/80.pkl')])['verified']


def test_rejected_second_plan_is_not_an_executed_plan():
    rows=[dict(kind='plan_accepted',step=66,target_id=0,request_id='r1'),
          dict(kind='execution_prediction',step=66,joint=dict(metadata=dict(anchor_guidance_active=True))),
          dict(kind='attack_execution',step=67,diffusion_source_step=66,target_id=0,request_id='r1',executed=True),
          dict(kind='plan_accepted',step=132,target_id=1,request_id='r2'),
          dict(kind='generation_failure',step=132)]
    result=plan_execution_ledger(rows)
    assert result['accepted_plans']==2 and result['executed_plans']==1 and result['unexecuted_plans']==1
    assert result['executed_plan_fraction']==.5 and not result['unmatched_execution_steps']


def test_explicit_request_id_mismatch_cannot_use_legacy_time_fallback():
    rows=[dict(kind='plan_accepted',step=0,target_id=0,request_id='r1'),
          dict(kind='attack_execution',step=1,target_id=0,request_id='wrong',executed=True)]
    assert plan_execution_ledger(rows)['unmatched_execution_steps']==[1]


def test_legacy_null_id_still_requires_matching_guidance_and_target():
    rows=[dict(kind='plan_accepted',step=0,target_id=0,request_id=None),
          dict(kind='execution_prediction',step=0,joint=dict(metadata=dict(anchor_guidance_active=False))),
          dict(kind='attack_execution',step=1,target_id=0,request_id=None,executed=True)]
    result=plan_execution_ledger(rows)
    assert result['executed_plans']==1 and result['execution_without_anchor_guidance_steps']==[1]


def test_measurement_decline_is_not_missing_template_coverage():
    rows=[dict(kind='llm_output',planner_trace=dict(context=dict(candidates=[],preflight_withheld_candidates=[{'candidate_id':'x0'}],
          measurement_preflight=dict(ready=False))))]
    result=catalog_ledger(rows)
    assert result['coverage_before_measurement']==1 and result['coverage_after_measurement']==0
    assert result['baseline_measurement_declines']==1


def test_per_scene_transport_limit_not_only_group_total():
    result=transport_ledger([dict(scenario_index=80) for _ in range(5)], [80,81],4)
    assert not result['per_scene_budget_verified']
    result=transport_ledger([dict(scenario_index=None)], [80],4)
    assert not result['per_scene_budget_verified'] and result['unknown_scene_metadata']==[None]


def test_missing_transport_log_is_unknown_not_zero_calls():
    result=transport_ledger([], [182],4,log_present=False,logical_calls=4)
    assert result['actual_transport_calls'] is None and result['total_tokens'] is None
    assert not result['per_scene_budget_verified']
