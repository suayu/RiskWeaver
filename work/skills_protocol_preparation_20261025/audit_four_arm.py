"""四组只读验收：中间消融组也必须完整，不能只检查新旧两组。"""
import json
from pathlib import Path
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0] = [str(ROOT/'work/skills_modelpool_budget_dev_20261022'),
               str(ROOT/'work/skills_recovery_v4_20261023')]
from pool_acceptance import audit_pool, paired_gate
from audit_recovery_v4 import component_configuration


def four_arm_gate(groups, expected, threshold=6):
    if not {'legacy_shared_recovery','skills_full_shared_recovery'} <= set(groups):
        return dict(passed=False, required_qualified_scenes=threshold, exact_group_coverage=False,
                    all_four_groups_integrity=False, reason='primary_groups_missing')
    paired = paired_gate(groups, threshold)
    exact = set(groups) == set(expected)
    integrity = exact and all(g.get('group_integrity_pass') for g in groups.values())
    return dict(passed=bool(paired['passed'] and integrity), required_qualified_scenes=threshold,
        legacy_qualified=paired['legacy_qualified'], skills_qualified=paired['skills_qualified'],
        all_four_groups_integrity=bool(integrity), exact_group_coverage=exact,
        scope='P1 development gate; not universal noninferiority, skill-text causality or goal completion')


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def declared_configuration(cfg, expected, receipt):
    """从协议读取每臂配置，不凭组名假定中间组必须关闭路线预检。"""
    return bool(cfg.sim.llm.skills.enabled == expected['skills_enabled']
        and cfg.sim.llm.skills.level == expected['level']
        and cfg.sim.llm.skills.candidate_expansion_enabled == expected['candidate_expansion']
        and cfg.sim.llm.skills.get('event_budget_enabled', False) == expected['skills_enabled']
        and receipt.get('incremental_skills') == ('on' if expected['skills_enabled'] else 'off')
        and receipt.get('active_window_recipes') == ('on' if expected['active_recipes'] else 'off')
        and receipt.get('bailian_capabilities') == 'observed')


def audit_four_arm(root):
    from omegaconf import OmegaConf
    result = audit_pool(root)
    launch = load(root/'launch.json')
    # 配置也必须来自实验源码快照，不使用可能已改变的工作目录元数据。
    relative = Path(launch['protocol']).relative_to(ROOT)
    protocol = load(root/'source_snapshot'/relative/'protocol.json')
    expected = protocol['arms']
    hashes = load(root/'source_sha256.json')
    for arm, group in result['groups'].items():
        config_path = root/arm/'resolved_config.yaml'
        recovery = load(root/arm/'recovery_v4_protocol.json')
        video = load(root/arm/'video_export_protocol.json')
        entry = load(root/arm/'incremental_entry_protocol.json')
        flags = expected.get(arm)
        cfg = OmegaConf.load(config_path) if config_path.exists() else None
        configuration_ok = bool(flags and cfg is not None and declared_configuration(cfg, flags, entry))
        if configuration_ok:
            # 复用原组件阈值，skills三臂均明确开启相同的路线尾部预检。
            expected_name = 'skills_full_shared_recovery' if flags['skills_enabled'] else 'legacy_shared_recovery'
            configuration_ok = component_configuration(cfg, recovery, video, expected_name, hashes)
            if flags['skills_enabled']:
                incremental = load(root/arm/'incremental_protocol.json')
                configuration_ok = bool(configuration_ok and incremental.get('enabled')
                    and incremental.get('extra_recipes') == flags['active_recipes']
                    and str(cfg.sim.llm.skills.root) == incremental.get('knowledge_root')
                    and incremental.get('source_sha256')
                    and all(hashes.get(n) == h for n, h in incremental['source_sha256'].items()))
        malformed = 0
        recoveries = []
        for trace in (root/arm).rglob('execution_trace.jsonl'):
            for line in trace.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if row.get('kind') == 'execution_prediction':
                    metadata = ((row.get('joint') or {}).get('metadata') or {}).get('temporal_joint_recovery')
                    if metadata:
                        recoveries.append(metadata)
        temporal_ok = all(m.get('recovered') and m.get('consecutive') == 1 and m.get('all_active_agents_verified')
            and m.get('no_extrapolation') and m.get('remaining_steps', 0) >= 30
            and m.get('additional_same_frame_diffusion_samples') == 0 and m.get('additional_LLM_calls') == 0
            and m.get('validation_seconds', float('inf')) <= 60. and m.get('area_gate', {}).get('allowed')
            for m in recoveries)
        group['next_protocol_configuration_verified'] = bool(configuration_ok)
        group['temporal_recovery_verified'] = temporal_ok and malformed == 0
        group['group_integrity_pass'] &= bool(configuration_ok and temporal_ok and malformed == 0)
    result['development_gate'] = four_arm_gate(result['groups'], expected)
    result['comparison_controls'] = protocol['comparison_controls']
    result['goal_complete'] = False
    return result


def smoke_prerequisite(root, arm, scenario_index):
    """先验证真实单场产物/新鲜调用/执行，不能仅凭退出码启动四十次尝试。"""
    from pool_acceptance import qualify_scene
    from audit_enhanced import plan_execution_ledger
    from experiments.skills.audit import audit_attempt
    final = load(root/'launcher_result.json')
    if final.get('exit_code') != 0:
        return dict(passed=False, reason='smoke_not_successful_terminal', experiment=str(root))
    files = list((root/arm).rglob('attempt_result.json'))
    if len(files) != 1:
        return dict(passed=False, reason='smoke_attempt_count', actual=len(files))
    attempt = audit_attempt(files[0])
    if attempt['scenario_index'] != scenario_index:
        return dict(passed=False, reason='wrong_smoke_scenario')
    rows = [json.loads(line) for line in files[0].with_name('execution_trace.jsonl').read_text().splitlines()]
    requests_path = root/f'{arm}_api_calls.jsonl'
    requests = [json.loads(line) for line in requests_path.read_text().splitlines()] if requests_path.exists() else []
    qualification = qualify_scene(attempt, plan_execution_ledger(rows), requests, 4)
    states = [r for r in rows if r.get('kind') == 'state']
    static_ok = bool(states) and all(r.get('static_obstacles') == [] for r in states)
    source = load(root/'runtime_source_integrity.json')
    source_ok = bool(source.get('verified') or (source.get('checks') and all(c.get('verified') for c in source['checks'])))
    unknown_requests = [r for r in requests if r.get('scenario_index') != scenario_index]
    return dict(passed=bool(qualification['qualified'] and static_ok and source_ok and not unknown_requests),
        scene=qualification, pure_dynamic_states_verified=static_ok, source_integrity_verified=source_ok,
        unknown_request_count=len(unknown_requests), scope='one fresh closed-loop scene; not majority coverage')
