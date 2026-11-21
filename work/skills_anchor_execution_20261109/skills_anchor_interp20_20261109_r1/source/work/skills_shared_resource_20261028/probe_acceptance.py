"""资源门禁不等于攻击性能验收；拒绝、超时及未执行必须完整披露。"""
from pathlib import Path
import json
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')


def load(path):return json.loads(path.read_text()) if path.exists() else None


def gate(checks):return bool(checks and all(value is True for value in checks.values()))


def assess(root,exit_code,stop_reason):
    resource=load(root/'probe_resource_result.json')
    path=root/'probe_sampling.jsonl';samples=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    memory=root/'probe_resource_memory.jsonl';memory_rows=[json.loads(x) for x in memory.read_text().splitlines()] if memory.exists() else []
    sys.path.insert(0,str(ROOT));from experiments.skills.audit import audit_attempt
    attempts=[audit_attempt(p) for p in sorted((root/'probe').rglob('attempt_result.json'))]
    sdk=root/'probe_api_calls.jsonl';physical=len(sdk.read_text().splitlines()) if sdk.exists() else 0
    successful=[x for x in samples if x.get('success')]
    timeout_reasons=[a.get('generation_failure_reason') for a in attempts if 'timeout' in str(a.get('generation_failure_reason','')).lower()]
    cap=14*2**30
    checks=dict(child_exit_zero=exit_code==0,no_resource_abort=stop_reason is None,
        both_frozen_scenes_recorded=sorted(a['scenario_index'] for a in attempts)==[53,74],
        no_runtime_failure=bool(attempts) and all(not a['runtime_failure'] for a in attempts),
        no_background_collision=bool(attempts) and all(not a['background_collision_frames'] and not a['background_static_collision_frames'] for a in attempts),
        no_static_obstacles=bool(attempts) and all(not a['static_obstacles_seen'] for a in attempts),
        all_video_and_trace_complete=bool(attempts) and all(a['completeness']['video_frames_match'] and a['completeness']['state_sequence_complete'] and a['completeness']['terminal_state_present'] for a in attempts),
        enough_guided_execution=sum(a['actual_attack_frames'] for a in attempts)>=10,
        at_least_one40frame_prefix=any(a['executed_steps']>=40 for a in attempts),
        both_source_scenes_sampled=len({s['scene_id'] for s in successful})>=2,
        original_eight_joint_samples=bool(samples) and all(s['num_samples']==8 for s in samples),
        guided_sampling_observed=any(s['anchor_active'] and s['success'] for s in samples),
        no_sampling_errors=bool(samples) and all(s['success'] for s in samples),
        sampling_within_resource_budget=bool(successful) and all(s['seconds']<=120 for s in successful),
        completed_memory_receipt=bool(resource and resource['error_type'] is None and resource['cap_gib']==14),
        allocator_within_cap=bool(resource and resource['peak_allocated_bytes']<=cap+2**20 and resource['peak_reserved_bytes']<=cap+2**20),
        memory_samples_present=bool(memory_rows),no_LLM_requests=physical==0)
    return dict(passed=gate(checks),scope='two_scene_shared_hardware_probe_only_not_method_performance',checks=checks,
        planned_scenes=2,attempted_scenes=len(attempts),source_indices=[a['scenario_index'] for a in attempts],
        generation_failures=[dict(scenario_index=a['scenario_index'],reason=a['generation_failure_reason']) for a in attempts if a['generation_failure']],
        method_timeout_reasons_retained=timeout_reasons,
        method_rejection_scope='retained_not_classified_as_GPU_memory_failure_or_method_success; no_wall_or_GPU_sampling_timeout_allowed',
        per_scene=[{k:a[k] for k in ['scenario_index','executed_steps','actual_attack_frames','collision','completed','progress','generation_failure','generation_failure_reason','runtime_failure','background_collision_frames','background_static_collision_frames','completeness']} for a in attempts],
        resource=resource,sampling_events=len(samples),successful_sampling_events=len(successful),
        source_scene_ids=sorted({s['scene_id'] for s in samples}),
        sampling_seconds=[s['seconds'] for s in samples],
        minimum_observed_free_bytes=min((m['free_bytes'] for m in memory_rows),default=None),
        physical_LLM_requests=physical,fixed_fixture_not_fresh_LLM=True,actual_attack_frames_not_comparison_evidence=True,
        no_continuous_safety_guarantee=True,goal_complete=False)
