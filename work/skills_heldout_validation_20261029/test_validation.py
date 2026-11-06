"""留出分母、启动门禁、参数保护和真实新入口识别回归，零API与GPU。"""
import copy
import json
import sys
import pytest
from validation_core import ARMS,INDICES,validate_manifest,rewrite_command,readiness,operational_gate


def manifest():return dict(scenario_count=10,original_scenario_indices=list(INDICES),
    scenario_files=[f'{i}.pkl' for i in range(10)],scenario_sha256=[f'{i:064x}' for i in range(10)])


def test_original_manifest_denominators_and_threshold():
    m=manifest();r=validate_manifest(m)
    assert r['planned_attempts']==20 and r['required_qualified_scenes']==6


@pytest.mark.parametrize('mutation',['twenty_scenes','reordered','duplicate_file','duplicate_hash'])
def test_wrong_scale_order_and_duplicate_sources_are_not_silently_replaced(mutation):
    m=manifest()
    if mutation=='twenty_scenes':m['scenario_count']=20
    elif mutation=='reordered':m['original_scenario_indices'].reverse()
    elif mutation=='duplicate_file':m['scenario_files'][1]=m['scenario_files'][0]
    else:m['scenario_sha256'][1]=m['scenario_sha256'][0]
    with pytest.raises(ValueError):validate_manifest(m)


def test_command_rewrite_preserves_solver_resource_and_method_flags():
    original=['bash','provider','dashscope','python','entry.py','--skills','on','--manifest','old.json',
        '--output-dir','old_output','--model','old_model','--models','old_model','another',
        '--policy','idm','--steps','400','--resource-cap-gib','14','--free-protection-confirmed']
    prior=copy.deepcopy(original);new=rewrite_command(original,'new.json','new_output',['a','b'])
    assert original==prior and new[new.index('--models')+1:new.index('--policy')]==['a','b']
    for k in ['--skills','--policy','--steps','--resource-cap-gib']:assert new[new.index(k)+1]==original[original.index(k)+1]
    assert new.count('--free-protection-confirmed')==1


def test_pending_smoke_blocks_even_when_development_passed():
    result=readiness(dict(passed=True),dict(passed=False,reason='not_terminal'),dict(verified=True),manifest())
    assert not result['passed'] and result['reasons']==['new_contract_smoke_not_passed_or_pending']


def test_source_drift_cannot_be_overridden_by_passed_smoke():
    result=readiness(dict(passed=True),dict(passed=True),dict(verified=False),manifest())
    assert not result['passed'] and 'source_or_algorithm_drift' in result['reasons']


def groups(old=4,new=6):return {ARMS[0]:dict(planned_scenes=10,attempted_scenes=10,qualified_scenes=old,group_integrity_pass=True),
    ARMS[1]:dict(planned_scenes=10,attempted_scenes=10,qualified_scenes=new,group_integrity_pass=True)}


def test_gate_retains_unattempted_denominator_and_no_broad_correctness_claim():
    g=groups();assert operational_gate(g)['passed']
    g[ARMS[1]]['attempted_scenes']=9
    r=operational_gate(g);assert not r['passed'] and r['planned_attempts']==20


def test_majority_alone_cannot_hide_worse_than_legacy_or_runtime_integrity():
    assert not operational_gate(groups(8,6))['passed']
    g=groups();g[ARMS[0]]['group_integrity_pass']=False
    assert not operational_gate(g)['passed']


def test_actual_launch_refused_before_any_output_when_gate_pending(tmp_path,monkeypatch):
    import run_validation
    protocol=tmp_path/'protocol';protocol.mkdir();(protocol/'protocol.json').write_text('{}')
    output=tmp_path/'experiment'
    monkeypatch.setattr(run_validation,'check_prerequisites',lambda *a:dict(passed=False,reasons=['pending']))
    monkeypatch.setattr(sys,'argv',['run','--protocol',str(protocol),'--output-dir',str(output)])
    with pytest.raises(ValueError,match='no_scenarios_dispatched'):run_validation.main()
    assert not output.exists()


def test_prepare_only_returns_pending_without_creating_experiment(tmp_path,monkeypatch):
    import run_validation
    protocol=tmp_path/'protocol';protocol.mkdir();(protocol/'protocol.json').write_text('{}')
    output=tmp_path/'experiment';monkeypatch.setattr(run_validation,'check_prerequisites',lambda *a:dict(passed=False,reasons=['pending']))
    monkeypatch.setattr(sys,'argv',['run','--protocol',str(protocol),'--output-dir',str(output),'--prepare-only'])
    assert not run_validation.main()['passed'] and not output.exists()


def receipts():
    common=dict(version='common_provider_control_20261028_v1',semantic_model_policy='service_only',
        thinking_required_exact_models=['glm-5.3','qwen3.8-2.4t-a95b'],true_service_fallback_unchanged=True,
        permanent_disable_only_404_or_explicit_quota=True,transport_limit_unchanged=True,
        skills_and_diffusion_and_safety_unchanged=True,extra_requests=0)
    resource=dict(version='shared_GPU_fence_v1_20261028',cap_gib=14,resource_fixture='off',
        sampling_synchronization_for_timing=False,only_own_torch_allocator_capped=True,error_type=None,
        source_solver_and_samples_unchanged=True)
    command=['python','work/skills_rolling_contract_20261029/run_contract_controlled.py',
        '--resource-cap-gib','14','--resource-reserve-gib','3','--resource-fixture','off','--resource-observe-sampling','off',
        '--semantic-model-policy','service_only','--rolling-intent','on','--explicit-contract','on','--deduplicate-intents','on']
    return command,common,resource,dict(enabled=True),dict(explicit_contract='on',deduplicate_intents='on',original_validator_preserved=True)


def test_actual_new_entry_cannot_be_replaced_by_old_entry_with_same_receipt():
    from audit_validation import entry_receipts_valid
    c,common,resource,rolling,contract=receipts()
    assert entry_receipts_valid(ARMS[1],c,common,resource,rolling,contract,'0')
    c[1]='work/skills_shared_resource_20261028/run_capped.py'
    assert not entry_receipts_valid(ARMS[1],c,common,resource,rolling,contract,'0')


def test_fixture_or_wrong_gpu_cannot_masquerade_as_real_model_evidence():
    from audit_validation import entry_receipts_valid
    c,common,resource,rolling,contract=receipts()
    assert not entry_receipts_valid(ARMS[1],c,common,resource,rolling,contract,'1')
    resource['resource_fixture']='on'
    assert not entry_receipts_valid(ARMS[1],c,common,resource,rolling,contract,'0')
