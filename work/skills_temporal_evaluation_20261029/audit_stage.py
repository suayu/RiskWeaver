"""独立时序版验收；保留原配对审计，再识别真实入口与运行trace。"""
from pathlib import Path
import json
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT/'work/skills_recovery_v4_20261023'),
    str(ROOT/'work/skills_provider_compat_20261028'),str(ROOT/'work/skills_heldout_validation_20261029')]
from audit_recovery_v4 import audit_v4
from audit_controlled import control_receipt_valid
from stage_core import ARMS,TEMPORAL_FLAGS,validate_manifest,trace_verified,stage_gate


def load(path):return json.loads(path.read_text()) if path.exists() else {}


def entry_verified(arm,command,common,resource,rolling,contract,temporal,gpu):
    def flag(k,v):return command.count(k)==1 and command.index(k)+1<len(command) and command[command.index(k)+1]==v
    entry='skills_shared_resource_20261028/run_capped.py' if arm==ARMS[0] else 'skills_temporal_candidates_20261029/run_temporal_controlled.py'
    common_ok=bool(arm in ARMS and any(str(x).endswith(entry) for x in command) and gpu=='0'
        and control_receipt_valid(common) and flag('--semantic-model-policy','service_only')
        and flag('--resource-cap-gib','14') and flag('--resource-reserve-gib','3')
        and flag('--resource-fixture','off') and flag('--resource-observe-sampling','off')
        and resource.get('version')=='shared_GPU_fence_v1_20261028' and resource.get('cap_gib')==14
        and resource.get('resource_fixture')=='off' and resource.get('sampling_synchronization_for_timing') is False
        and resource.get('only_own_torch_allocator_capped') is True and resource.get('error_type') is None
        and resource.get('source_solver_and_samples_unchanged') is True)
    if arm==ARMS[1]:
        common_ok &= bool(all(flag('--'+k.replace('_','-'),'on') for k in TEMPORAL_FLAGS)
            and flag('--rolling-intent','on') and flag('--explicit-contract','on') and flag('--deduplicate-intents','on')
            and rolling.get('enabled') is True and contract.get('explicit_contract')=='on'
            and contract.get('deduplicate_intents')=='on' and contract.get('original_validator_preserved') is True
            and temporal.get('version')=='temporal_entry_20261029_v1'
            and all(temporal.get('flags',{}).get(k) is True for k in TEMPORAL_FLAGS)
            and temporal.get('interaction_radius_m')==15.
            and temporal.get('other_validation_and_diffusion_unchanged') is True)
    return common_ok


def audit_stage(root):
    launch=load(root/'launch.json');stage=launch.get('stage')
    validate_manifest(load(root/'manifest.json'),stage)
    result=audit_v4(root);hashes=load(root/'source_sha256.json')
    for arm,group in result['groups'].items():
        pid=load(root/f'{arm}_pid.json')
        entry=entry_verified(arm,pid.get('command',[]),load(root/arm/'common_provider_control.json'),
            load(root/f'{arm}_resource_result.json'),load(root/arm/'rolling_entry_protocol.json'),
            load(root/arm/'explicit_contract_entry.json'),load(root/arm/'temporal_entry_protocol.json'),pid.get('gpu'))
        states=[];traces=[];contract=[]
        for path in (root/arm).rglob('execution_trace.jsonl'):
            for row in map(json.loads,path.read_text().splitlines()):
                if row.get('kind')=='state':states.append(row)
                planner=row.get('planner_trace') or {}
                if row.get('kind')=='llm_output' and planner.get('requests'):
                    traces.append(planner.get('temporal_candidate_protocol') or {})
                    contract.append(planner.get('explicit_contract_protocol') or {})
        pure=bool(states) and all(r.get('static_obstacles')==[] for r in states)
        actual=True
        if arm==ARMS[1]:
            contract_sha=hashes.get('work/skills_rolling_contract_20261029/contract_planner.py')
            actual=bool(traces and contract_sha) and all(trace_verified(t,hashes) for t in traces)
            actual &= all(t.get('explicit_contract') is True and t.get('numeric_deduplication') is True
                and t.get('source_sha256')==contract_sha and t.get('original_response_validator_preserved') is True
                and t.get('response_arrays_not_silently_converted') is True for t in contract)
        group['temporal_entry_and_resource_verified']=entry
        group['actual_temporal_factory_trace_verified']=actual
        group['pure_dynamic_states_verified']=pure
        group['group_integrity_pass'] &= bool(entry and actual and pure)
    result['development_gate']=stage_gate(result['groups'])
    result['stage']=stage;result['goal_complete']=False
    result['semantic_review_status']='required_not_automatically_certified'
    result['scope']='whole_temporal_package_not_skill_text_only'
    return result
