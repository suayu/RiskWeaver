"""共享资源四组验收：保留原四组门禁，并核对同一显存围栏与真实LLM路径。"""
from pathlib import Path
import json
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT/'work/skills_protocol_preparation_20261025'),str(ROOT/'work/skills_provider_compat_20261028')]
from audit_four_arm import audit_four_arm
from audit_controlled import control_receipt_valid


def shared_receipts_valid(command,common,resource,gpu):
    def flag(name,value):return name in command and command[command.index(name)+1]==value
    return bool(control_receipt_valid(common)
        and any(str(v).endswith('/skills_shared_resource_20261028/run_capped.py') for v in command)
        and flag('--semantic-model-policy','service_only')
        and flag('--resource-cap-gib','14') and flag('--resource-reserve-gib','3')
        and flag('--resource-fixture','off') and flag('--resource-observe-sampling','off')
        and flag('--steps','400') and gpu=='0'
        and resource.get('version')=='shared_GPU_fence_v1_20261028'
        and resource.get('cap_gib')==14 and resource.get('resource_fixture')=='off'
        and resource.get('source_solver_and_samples_unchanged') is True
        and resource.get('sampling_synchronization_for_timing') is False
        and resource.get('only_own_torch_allocator_capped') is True
        and resource.get('error_type') is None)


def audit_shared(root):
    result=audit_four_arm(root)
    for arm,group in result['groups'].items():
        p=root/(arm+'_pid.json');pid=json.loads(p.read_text()) if p.exists() else {}
        p=root/arm/'common_provider_control.json';common=json.loads(p.read_text()) if p.exists() else {}
        p=root/(arm+'_resource_result.json');resource=json.loads(p.read_text()) if p.exists() else {}
        check=shared_receipts_valid(pid.get('command',[]),common,resource,pid.get('gpu'))
        group['shared_resource_protocol_verified']=check
        group['group_integrity_pass'] &= check
    integrity=all(g['group_integrity_pass'] for g in result['groups'].values())
    result['development_gate']['all_four_groups_integrity'] &= integrity
    result['development_gate']['passed'] &= integrity
    result['shared_resource_scope']='same_GPU0_single_arm14GiB_cap; no_fixture_or_extra_sampling_sync; competing_workloads_disclosed'
    return result
