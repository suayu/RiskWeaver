"""时序版开发/留出阶段的版本隔离；纯函数，不发API、不运行仿真。"""
import re

ARMS=('legacy_shared_recovery','skills_full_shared_recovery')
INDICES={
    'development':(53,56,74,80,103,140,181,182,207,220),
    'validation':(6,28,45,102,114,143,146,185,229,232),
}
TEMPORAL_FLAGS=('fractional_interaction','velocity_continuation','motion_evidence')
TEMPORAL_SOURCE='work/skills_temporal_candidates_20261029/temporal_planner.py'
RUNTIME_SOURCES=tuple('work/skills_temporal_candidates_20261029/'+name for name in
    ('temporal_planner.py','temporal_intents.py','run_temporal_controlled.py'))


def validate_manifest(manifest,stage):
    """保持原场景、顺序、分母；开发场景不能冒充独立留出。"""
    if stage not in INDICES or manifest.get('scenario_count')!=10 or tuple(manifest.get('original_scenario_indices',[]))!=INDICES[stage]:
        raise ValueError('wrong_original_stage_manifest')
    files=manifest.get('scenario_files',[]);hashes=manifest.get('scenario_sha256',[])
    if len(files)!=10 or len(set(files))!=10 or len(hashes)!=10 or len(set(hashes))!=10:
        raise ValueError('duplicate_or_missing_scenario_source')
    if any(not isinstance(h,str) or not re.fullmatch('[a-f0-9]{64}',h) for h in hashes):
        raise ValueError('invalid_scenario_sha256')
    return dict(planned_attempts=20,planned_scenes_per_arm=10,required_qualified_scenes=6)


def same_temporal_runtime(previous,current):
    """比较共享依赖，不只看新入口；阶段特有协议不相交，模型注释由公共审计负责。"""
    required=all(previous.get(n) and previous.get(n)==current.get(n) for n in RUNTIME_SOURCES)
    shared=set(previous)&set(current)-{'policies/llm_adversarial_planner.py'}
    return required and all(previous[n]==current[n] for n in shared)


def trace_verified(trace,hashes):
    return bool(trace.get('version')=='temporal_candidate_prototype_20261029_v1'
        and all(trace.get(k) is True for k in TEMPORAL_FLAGS)
        and hashes.get(TEMPORAL_SOURCE) and trace.get('source_sha256')==hashes[TEMPORAL_SOURCE]
        and trace.get('semantic_interaction_radius_m')==15.
        and trace.get('background_dynamics_and_diffusion_unchanged') is True
        and trace.get('original_off_path_preserved') is True)


def stage_gate(groups):
    if set(groups)!=set(ARMS):return dict(passed=False,reason='exact_two_arm_set_required')
    old,new=(groups[n] for n in ARMS)
    exact=all(g.get('planned_scenes')==g.get('attempted_scenes')==10 for g in groups.values())
    integrity=all(g.get('group_integrity_pass') is True for g in groups.values())
    return dict(passed=bool(exact and integrity and new.get('qualified_scenes',0)>=6
        and new.get('qualified_scenes',0)>=old.get('qualified_scenes',0)),
        planned_attempts=20,required_qualified_scenes=6,all_scheduled_attempts_in_denominator=True,
        legacy_qualified=old.get('qualified_scenes'),skills_qualified=new.get('qualified_scenes'),
        exact_coverage=exact,all_groups_integrity=integrity,
        scope='operational_majority_not_semantic_correctness_or_threat_certificate')


def readiness(stage,smoke,source,stage_manifest,temporal_match,development=None):
    counts=validate_manifest(stage_manifest,stage);reasons=[]
    if not smoke.get('passed'):reasons.append('current_temporal_smoke_not_passed_or_pending')
    if not source.get('verified'):reasons.append('current_source_drift')
    if not temporal_match:reasons.append('smoke_runtime_version_mismatch')
    if stage=='validation':
        # 旧四组门禁不能替代新时序版十场开发；单场成功也不能替代多数覆盖。
        dev=development or {}
        if not dev.get('terminal_completed'):reasons.append('current_development_not_completed')
        if not dev.get('current_runtime_verified'):reasons.append('development_runtime_version_mismatch')
        if not dev.get('gate',{}).get('passed'):reasons.append('current_development_gate_not_passed')
        if not dev.get('original_development_manifest_verified'):reasons.append('development_manifest_not_verified')
    return dict(passed=not reasons,reasons=reasons,stage=stage,**counts,goal_complete=False,
        semantic_correctness_and_actual_threat_require_independent_review=True)
