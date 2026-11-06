"""保留原配对、视频、时域恢复和预算验收，额外检查新入口及共享围栏。"""
from pathlib import Path
import json
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT/'work/skills_recovery_v4_20261023'),str(ROOT/'work/skills_provider_compat_20261028')]
from audit_recovery_v4 import audit_v4
from audit_controlled import control_receipt_valid
from validation_core import ARMS,validate_manifest,operational_gate


def load(path):return json.loads(path.read_text()) if path.exists() else {}


def entry_receipts_valid(arm,command,common,resource,rolling,contract,gpu):
    def flag(name,value):return command.count(name)==1 and command[command.index(name)+1]==value
    entry='skills_shared_resource_20261028/run_capped.py' if arm==ARMS[0] else 'skills_rolling_contract_20261029/run_contract_controlled.py'
    checks=bool(arm in ARMS and any(str(v).endswith(entry) for v in command)
        and control_receipt_valid(common) and gpu=='0'
        and flag('--resource-cap-gib','14') and flag('--resource-reserve-gib','3')
        and flag('--resource-fixture','off') and flag('--resource-observe-sampling','off')
        and flag('--semantic-model-policy','service_only')
        and resource.get('version')=='shared_GPU_fence_v1_20261028' and resource.get('cap_gib')==14
        and resource.get('resource_fixture')=='off' and resource.get('sampling_synchronization_for_timing') is False
        and resource.get('only_own_torch_allocator_capped') is True and resource.get('error_type') is None
        and resource.get('source_solver_and_samples_unchanged') is True)
    if arm==ARMS[1]:
        checks &= bool(flag('--rolling-intent','on') and flag('--explicit-contract','on') and flag('--deduplicate-intents','on')
            and rolling.get('enabled') is True and contract.get('explicit_contract')=='on'
            and contract.get('deduplicate_intents')=='on' and contract.get('original_validator_preserved') is True)
    return checks


def audit_validation(root):
    validate_manifest(load(root/'manifest.json'));result=audit_v4(root)
    for arm,group in result['groups'].items():
        pid=load(root/f'{arm}_pid.json')
        verified=entry_receipts_valid(arm,pid.get('command',[]),load(root/arm/'common_provider_control.json'),
            load(root/f'{arm}_resource_result.json'),load(root/arm/'rolling_entry_protocol.json'),
            load(root/arm/'explicit_contract_entry.json'),pid.get('gpu'))
        states=[];contract_traces=[]
        for trace in (root/arm).rglob('execution_trace.jsonl'):
            for row in map(json.loads,trace.read_text().splitlines()):
                if row.get('kind')=='state':states.append(row)
                if row.get('kind')=='llm_output' and (row.get('planner_trace') or {}).get('requests'):
                    contract_traces.append((row.get('planner_trace') or {}).get('explicit_contract_protocol') or {})
        pure_dynamic=bool(states) and all(r.get('static_obstacles')==[] for r in states)
        actual_factory=True
        if arm==ARMS[1]:
            hashes=load(root/'source_sha256.json');expected=hashes.get('work/skills_rolling_contract_20261029/contract_planner.py')
            actual_factory=bool(expected and contract_traces) and all(t.get('explicit_contract') is True
                and t.get('numeric_deduplication') is True and t.get('source_sha256')==expected
                and t.get('original_response_validator_preserved') is True
                and t.get('response_arrays_not_silently_converted') is True for t in contract_traces)
        group['heldout_entry_and_shared_protocol_verified']=verified
        group['pure_dynamic_states_verified']=pure_dynamic
        group['actual_contract_factory_trace_verified']=actual_factory
        group['group_integrity_pass'] &= bool(verified and pure_dynamic and actual_factory)
    result['validation_gate']=operational_gate(result['groups'])
    result['development_gate']=result['validation_gate']  # 兼容旧启动器的输出字段，语义明确为本次运行门槛。
    result['goal_complete']=False
    result['semantic_review_status']='required_not_automatically_certified'
    result['scope']='legacy_vs_complete_rolling_contract_package; not_skill_text_only_effect'
    return result
