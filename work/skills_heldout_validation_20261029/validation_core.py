"""原留出10场的纯配置/门禁函数；不接触API、GPU或仿真状态。"""
import copy
import re

ARMS=('legacy_shared_recovery','skills_full_shared_recovery')
INDICES=(6,28,45,102,114,143,146,185,229,232)


def validate_manifest(manifest):
    """此入口仅服务已约定留出10场；不能把6/10门槛误用于其他规模。"""
    if manifest.get('scenario_count')!=10 or tuple(manifest.get('original_scenario_indices',[]))!=INDICES:
        raise ValueError('original_validation10_manifest_required')
    files=manifest.get('scenario_files',[]);hashes=manifest.get('scenario_sha256',[])
    if len(files)!=10 or len(set(files))!=10 or len(hashes)!=10 or len(set(hashes))!=10:
        raise ValueError('manifest_file_hash_count_or_duplicates')
    if any(not isinstance(h,str) or not re.fullmatch('[a-f0-9]{64}',h) for h in hashes):raise ValueError('invalid_content_hash')
    return dict(planned_scenes_per_arm=10,planned_attempts=20,required_qualified_scenes=6,
        exposure_scope='not_used_for_this_skills_development_not_certified_project_unseen')


def rewrite_command(command,manifest,output,pool):
    """只替换本阶段数据/输出与共同批准模型池，不改方法/求解器/资源/安全开关。"""
    if not pool or len(pool)!=len(set(pool)):raise ValueError('empty_or_duplicate_model_pool')
    cmd=list(command)
    for key,value in [('--manifest',manifest),('--output-dir',output),('--model',pool[0])]:
        if cmd.count(key)!=1:raise ValueError('ambiguous_command_flag:'+key)
        cmd[cmd.index(key)+1]=str(value)
    if cmd.count('--models')!=1:raise ValueError('ambiguous_model_list')
    first=cmd.index('--models')+1;last=first
    while last<len(cmd) and not cmd[last].startswith('--'):last+=1
    cmd[first:last]=pool
    if '--free-protection-confirmed' not in cmd:cmd.append('--free-protection-confirmed')
    if cmd.count('--free-protection-confirmed')!=1:raise ValueError('duplicate_confirmation_marker')
    return cmd


def readiness(development_gate,smoke,source_check,manifest):
    counts=validate_manifest(manifest);reasons=[]
    if not development_gate.get('passed'):reasons.append('four_arm_development_gate_not_passed')
    if not smoke.get('passed'):reasons.append('new_contract_smoke_not_passed_or_pending')
    if not source_check.get('verified'):reasons.append('source_or_algorithm_drift')
    return dict(passed=not reasons,reasons=reasons,**counts,not_total_goal_completion=True,
        correct_tactics_and_actual_threat_require_separate_review=True)


def operational_gate(groups):
    if set(groups)!=set(ARMS):return dict(passed=False,reason='exact_paired_arm_set_required')
    old,new=(groups[a] for a in ARMS)
    exact=all(g.get('planned_scenes')==g.get('attempted_scenes')==10 for g in groups.values())
    integrity=all(g.get('group_integrity_pass') for g in groups.values())
    return dict(passed=bool(exact and integrity and new['qualified_scenes']>=6 and new['qualified_scenes']>=old['qualified_scenes']),
        planned_attempts=20,all_scheduled_attempts_in_denominator=True,required_qualified_scenes=6,
        legacy_qualified=old.get('qualified_scenes'),skills_qualified=new.get('qualified_scenes'),
        exact_coverage=exact,all_groups_integrity=integrity,
        scope='heldout_operational_majority_not_semantic_correctness_threat_noninferiority_or_goal_completion')
