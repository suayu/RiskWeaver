"""在原四组验收之上核对公共提供方修复，不能因新入口跳过旧门禁。"""
import json
from pathlib import Path
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0, str(ROOT/'work/skills_protocol_preparation_20261025'))
from audit_four_arm import audit_four_arm


def control_receipt_valid(receipt):
    return bool(receipt.get('version') == 'common_provider_control_20261028_v1'
        and receipt.get('semantic_model_policy') == 'service_only'
        and receipt.get('thinking_required_exact_models') == ['glm-5.3', 'qwen3.8-2.4t-a95b']
        and receipt.get('true_service_fallback_unchanged') is True
        and receipt.get('permanent_disable_only_404_or_explicit_quota') is True
        and receipt.get('transport_limit_unchanged') is True
        and receipt.get('skills_and_diffusion_and_safety_unchanged') is True
        and receipt.get('extra_requests') == 0)


def audit_controlled(root):
    result = audit_four_arm(root)
    for arm, group in result['groups'].items():
        p = root/arm/'common_provider_control.json'
        receipt = json.loads(p.read_text()) if p.exists() else {}
        pid_path = root/(arm+'_pid.json')
        command = json.loads(pid_path.read_text()).get('command', []) if pid_path.exists() else []
        # 既核对收据也核对真实派发命令；旧入口不能冒充公共适配已启用。
        command_ok = bool('--semantic-model-policy' in command
            and command[command.index('--semantic-model-policy')+1] == 'service_only'
            and any(str(x).endswith('/skills_provider_compat_20261028/run_controlled.py') for x in command))
        group['common_provider_control_verified'] = control_receipt_valid(receipt) and command_ok
        group['group_integrity_pass'] &= group['common_provider_control_verified']
    integrity = all(g['group_integrity_pass'] for g in result['groups'].values())
    result['development_gate']['all_four_groups_integrity'] &= integrity
    result['development_gate']['passed'] &= integrity
    result['common_provider_protocol'] = 'parameter_fix_and_semantic_retention; service_fallback_and_budget_preserved'
    return result
