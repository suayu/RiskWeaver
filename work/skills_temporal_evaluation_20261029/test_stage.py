"""阶段隔离、真实入口和未尝试分母回归；不调用API或GPU。"""
import copy
import json
import sys
import pytest
from stage_core import ARMS,INDICES,TEMPORAL_FLAGS,RUNTIME_SOURCES,TEMPORAL_SOURCE
from stage_core import validate_manifest,same_temporal_runtime,trace_verified,readiness,stage_gate


def manifest(stage='development'):
    return dict(scenario_count=10,original_scenario_indices=list(INDICES[stage]),
        scenario_files=[f'{stage}_{i}.pkl' for i in range(10)],scenario_sha256=[f'{i:064x}' for i in range(10)])


@pytest.mark.parametrize('stage',['development','validation'])
def test_original_stage_denominator(stage):
    assert validate_manifest(manifest(stage),stage)['planned_attempts']==20


def test_development_manifest_cannot_be_called_holdout():
    with pytest.raises(ValueError):validate_manifest(manifest(),'validation')


@pytest.mark.parametrize('mutation',['reorder','duplicate','count'])
def test_no_order_scale_or_source_substitution(mutation):
    m=manifest()
    if mutation=='reorder':m['original_scenario_indices'].reverse()
    elif mutation=='duplicate':m['scenario_sha256'][1]=m['scenario_sha256'][0]
    else:m['scenario_count']=20
    with pytest.raises(ValueError):validate_manifest(m,'development')


def test_equal_file_count_not_equivalent_to_equal_runtime():
    old={n:'a' for n in RUNTIME_SOURCES};new=copy.deepcopy(old)
    assert same_temporal_runtime(old,new)
    new[RUNTIME_SOURCES[1]]='b';assert not same_temporal_runtime(old,new)
    assert not same_temporal_runtime({}, {})


def test_inherited_code_drift_is_not_hidden_by_unchanged_temporal_class():
    old={n:'a' for n in RUNTIME_SOURCES};old['policies/skills_planner.py']='old'
    new=dict(old);new['policies/skills_planner.py']='new'
    assert not same_temporal_runtime(old,new)


def test_development_pending_smoke_blocks_even_with_old_development_pass():
    result=readiness('development',dict(passed=False),dict(verified=True),manifest(),True,dict(gate=dict(passed=True)))
    assert not result['passed'] and result['reasons']==['current_temporal_smoke_not_passed_or_pending']


def test_prior_contract_smoke_cannot_certify_temporal_version():
    result=readiness('development',dict(passed=True),dict(verified=True),manifest(),False)
    assert not result['passed'] and 'smoke_runtime_version_mismatch' in result['reasons']


def test_single_scene_and_old_four_arm_gate_cannot_start_new_holdout():
    old=dict(terminal_completed=True,gate=dict(passed=True),original_development_manifest_verified=True)
    result=readiness('validation',dict(passed=True),dict(verified=True),manifest('validation'),True,old)
    assert not result['passed'] and result['reasons']==['development_runtime_version_mismatch']


def test_fresh_development_is_required_to_be_terminal_and_original():
    dev=dict(terminal_completed=False,current_runtime_verified=True,gate=dict(passed=True),original_development_manifest_verified=False)
    result=readiness('validation',dict(passed=True),dict(verified=True),manifest('validation'),True,dev)
    assert not result['passed'] and set(result['reasons'])=={'current_development_not_completed','development_manifest_not_verified'}


def test_no_automatic_goal_or_correctness_claim_even_when_operationally_ready():
    dev=dict(terminal_completed=True,current_runtime_verified=True,gate=dict(passed=True),original_development_manifest_verified=True)
    result=readiness('validation',dict(passed=True),dict(verified=True),manifest('validation'),True,dev)
    assert result['passed'] and result['goal_complete'] is False
    assert result['semantic_correctness_and_actual_threat_require_independent_review']


def groups(old=4,new=6):return {n:dict(planned_scenes=10,attempted_scenes=10,qualified_scenes=k,group_integrity_pass=True)
    for n,k in zip(ARMS,[old,new])}


def test_all_unattempted_remain_in_denominator():
    g=groups();assert stage_gate(g)['passed'];g[ARMS[1]]['attempted_scenes']=9
    assert not stage_gate(g)['passed'] and stage_gate(g)['planned_attempts']==20


def test_majority_does_not_override_worse_than_old_or_integrity():
    assert not stage_gate(groups(8,6))['passed']
    g=groups();g[ARMS[0]]['group_integrity_pass']=False;assert not stage_gate(g)['passed']


def test_planner_factory_trace_not_just_declared_config():
    h={TEMPORAL_SOURCE:'a'};t=dict(version='temporal_candidate_prototype_20261029_v1',
        **dict.fromkeys(TEMPORAL_FLAGS,True),source_sha256='a',semantic_interaction_radius_m=15.,
        background_dynamics_and_diffusion_unchanged=True,original_off_path_preserved=True)
    assert trace_verified(t,h)
    t['velocity_continuation']=False;assert not trace_verified(t,h)
    t['velocity_continuation']=True;t['source_sha256']='b';assert not trace_verified(t,h)


def receipts():
    common=dict(version='common_provider_control_20261028_v1',semantic_model_policy='service_only',
        thinking_required_exact_models=['glm-5.3','qwen3.8-2.4t-a95b'],true_service_fallback_unchanged=True,
        permanent_disable_only_404_or_explicit_quota=True,transport_limit_unchanged=True,
        skills_and_diffusion_and_safety_unchanged=True,extra_requests=0)
    resource=dict(version='shared_GPU_fence_v1_20261028',cap_gib=14,resource_fixture='off',
        sampling_synchronization_for_timing=False,only_own_torch_allocator_capped=True,error_type=None,source_solver_and_samples_unchanged=True)
    command=['python','work/skills_temporal_candidates_20261029/run_temporal_controlled.py',
        '--resource-cap-gib','14','--resource-reserve-gib','3','--resource-fixture','off','--resource-observe-sampling','off',
        '--semantic-model-policy','service_only','--rolling-intent','on','--explicit-contract','on','--deduplicate-intents','on']
    for k in TEMPORAL_FLAGS:command+=['--'+k.replace('_','-'),'on']
    temporal=dict(version='temporal_entry_20261029_v1',flags=dict.fromkeys(TEMPORAL_FLAGS,True),
        interaction_radius_m=15.,other_validation_and_diffusion_unchanged=True)
    return command,common,resource,dict(enabled=True),dict(explicit_contract='on',deduplicate_intents='on',original_validator_preserved=True),temporal


def test_old_contract_entry_is_not_accepted_as_new_temporal():
    from audit_stage import entry_verified
    c,*rest=receipts();assert entry_verified(ARMS[1],c,*rest,'0')
    c[1]='work/skills_rolling_contract_20261029/run_contract_controlled.py'
    assert not entry_verified(ARMS[1],c,*rest,'0')


@pytest.mark.parametrize('problem',['fixture','wrong_gpu','flag_off','duplicate_flag'])
def test_runtime_receipt_cannot_hide_disabled_mode_or_resource_fixture(problem):
    from audit_stage import entry_verified
    c,common,resource,rolling,contract,temporal=receipts();gpu='0'
    if problem=='fixture':resource['resource_fixture']='on'
    elif problem=='wrong_gpu':gpu='1'
    elif problem=='flag_off':c[c.index('--velocity-continuation')+1]='off'
    else:c+=['--motion-evidence','on']
    assert not entry_verified(ARMS[1],c,common,resource,rolling,contract,temporal,gpu)


def test_real_dispatch_branch_refuses_pending_before_output(tmp_path,monkeypatch):
    import run_stage
    protocol=tmp_path/'protocol';protocol.mkdir();(protocol/'protocol.json').write_text('{}')
    target=tmp_path/'out';monkeypatch.setattr(run_stage,'check_prerequisites',lambda *a:dict(passed=False))
    monkeypatch.setattr(sys,'argv',['run','--protocol',str(protocol),'--output-dir',str(target)])
    with pytest.raises(ValueError,match='no_scenarios_dispatched'):run_stage.main()
    assert not target.exists()


def test_prepare_only_real_entry_never_creates_output_when_pending(tmp_path,monkeypatch):
    import run_stage
    protocol=tmp_path/'protocol';protocol.mkdir();(protocol/'protocol.json').write_text('{}')
    target=tmp_path/'out';monkeypatch.setattr(run_stage,'check_prerequisites',lambda *a:dict(passed=False))
    monkeypatch.setattr(sys,'argv',['run','--protocol',str(protocol),'--output-dir',str(target),'--prepare-only'])
    assert not run_stage.main()['passed'] and not target.exists()
