"""新门禁只能在原验收基础上收紧，不能绕过失败或原始配置校验。"""
from pathlib import Path
import copy
import json

import audit_controlled as audit


def receipt():
    return dict(version='common_provider_control_20261028_v1',
        semantic_model_policy='service_only',
        thinking_required_exact_models=['glm-5.3','qwen3.8-2.4t-a95b'],
        true_service_fallback_unchanged=True, permanent_disable_only_404_or_explicit_quota=True,
        transport_limit_unchanged=True, skills_and_diffusion_and_safety_unchanged=True, extra_requests=0)


def fixture(tmp_path, monkeypatch, prior_pass=True):
    groups={a:dict(group_integrity_pass=True) for a in ['legacy_shared_recovery','catalog_no_knowledge','skills_no_expansion','skills_full_shared_recovery']}
    base=dict(groups=groups, development_gate=dict(passed=prior_pass,all_four_groups_integrity=True))
    monkeypatch.setattr(audit,'audit_four_arm',lambda root:copy.deepcopy(base))
    for arm in groups:
        (tmp_path/arm).mkdir()
        (tmp_path/arm/'common_provider_control.json').write_text(json.dumps(receipt()))
        (tmp_path/(arm+'_pid.json')).write_text(json.dumps(dict(command=['python','/home2/zhaoyx/scenario-dreamer/work/skills_provider_compat_20261028/run_controlled.py','--semantic-model-policy','service_only'])))
    return groups


def test_complete_receipts_never_turn_original_failure_into_success(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,False)
    assert audit.audit_controlled(tmp_path)['development_gate']['passed'] is False


def test_missing_receipt_blocks_whole_gate(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    # 模拟未写出收据的失败，不让缺失记录被当作默认启用。
    (tmp_path/'catalog_no_knowledge/common_provider_control.json').rename(tmp_path/'catalog_no_knowledge/missing_receipt.json')
    result=audit.audit_controlled(tmp_path)
    assert not result['development_gate']['passed']
    assert not result['groups']['catalog_no_knowledge']['common_provider_control_verified']


def test_old_command_cannot_claim_new_control_receipt(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    (tmp_path/'legacy_shared_recovery_pid.json').write_text(json.dumps(dict(command=['python','old.py'])))
    assert not audit.audit_controlled(tmp_path)['development_gate']['passed']


def test_all_valid_keeps_original_gate(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    result=audit.audit_controlled(tmp_path)
    assert result['development_gate']['passed']
    assert all(g['common_provider_control_verified'] for g in result['groups'].values())
