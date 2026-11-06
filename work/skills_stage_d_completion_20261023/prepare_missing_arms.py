"""准备原四臂闭环消融的缺失两臂；不发API、不运行仿真、不改运行中源码。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


ARMS = {
    'legacy_free': dict(reuse='legacy_shared_recovery', skills='off', level=None, expansion=None),
    'unified_catalog_no_knowledge': dict(reuse=None, skills='on', level='none', expansion='off'),
    'skills_no_expansion': dict(reuse=None, skills='on', level='full', expansion='off'),
    'skills_with_expansion': dict(reuse='skills_full_shared_recovery', skills='on', level='full', expansion='on'),
}
COMMON = {
    '--policy': 'idm', '--seed': '42', '--steps': '400', '--transport-limit': '4',
    '--joint-candidate-recovery': 'on', '--device': 'cuda', '--video-export': 'integer',
    '--execution-area-policy': 'positive_area', '--temporal-recovery': 'on',
    '--temporal-validation-seconds': '60', '--group': 'riskweaver_profile_off',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(command, flag):
    """重复参数会隐含最后值覆盖，准备阶段必须拒绝这种不明确配置。"""
    if command.count(flag) != 1:
        raise ValueError('missing_or_duplicate_flag:' + flag)
    index = command.index(flag)
    if index + 1 >= len(command) or command[index + 1].startswith('--'):
        raise ValueError('missing_flag_value:' + flag)
    return command[index + 1]


def replace(command, flag, selected):
    value(command, flag)
    command[command.index(flag) + 1] = selected


def missing_command(template, arm, manifest, output):
    if arm not in ARMS or ARMS[arm]['reuse']:
        raise ValueError('not_a_missing_arm')
    command = list(template)
    for flag, selected in COMMON.items():
        if value(command, flag) != selected:
            raise ValueError('common_control_drift:' + flag)
    for flag in ('--skills', '--compact-view', '--event-budget', '--reachability-preflight', '--route-tail-preflight'):
        if value(command, flag) != 'on':
            raise ValueError('structured_arm_feature_drift:' + flag)
    if value(command, '--max-screening-queries') != '40':
        raise ValueError('screening_budget_drift')
    replace(command, '--skill-level', ARMS[arm]['level'])
    replace(command, '--candidate-expansion', ARMS[arm]['expansion'])
    replace(command, '--manifest', str(manifest))
    replace(command, '--output-dir', str(output))
    # 仍须由后续运行器检查空闲GPU、源码与免费保护；不能直接视为启动授权收据。
    if '--free-protection-confirmed' in command or '--prepare-only' in command:
        raise ValueError('runtime_or_prepare_flag_in_template')
    return command


def prepare(root, protocol, predecessor, destination, future_output):
    if destination.exists() or future_output.exists():
        raise FileExistsError('preserve_existing_outputs')
    terminal = read(predecessor / 'launcher_result.json')
    if terminal.get('status') != 'completed' or terminal.get('acceptance_passed') is not True:
        raise ValueError('development_predecessor_not_qualified')
    expected = read(protocol / 'development10.json')
    if read(predecessor / 'manifest.json') != expected:
        raise ValueError('development_manifest_mismatch')
    validation = read(protocol / 'validation10.json')
    if set(expected['original_scenario_indices']) & set(validation['original_scenario_indices']):
        raise ValueError('development_validation_overlap')
    if len(expected['original_scenario_indices']) != 10:
        raise ValueError('requires_original_development10')
    for path, sha in zip(expected['scenario_files'], expected['scenario_sha256']):
        if digest(Path(path)) != sha:
            raise ValueError('scene_input_changed')
    # 运行期允许模型列表仅增加停用注释，但准备文件必须忠实记录此刻源码，不能恢复耗尽模型。
    frozen = read(protocol / 'source_sha256.json')
    for relative, sha in frozen.items():
        if relative == 'policies/llm_adversarial_planner.py':
            continue
        if digest(root / relative) != sha:
            raise ValueError('source_drift:' + relative)
    sys.path.insert(0, str(root / 'work/skills_modelpool_budget_dev_20261022'))
    from model_pool_protocol import inspect_model_source, verify_pool_transition
    pool_before = read(protocol / 'protocol.json')['model_source']
    pool_now = inspect_model_source(root / 'policies/llm_adversarial_planner.py')
    pool_transition = verify_pool_transition(pool_before, pool_now)
    if not pool_transition['verified']:
        raise ValueError('model_source_algorithm_or_pool_drift')
    receipts = read(predecessor / 'runtime_source_integrity.json')['checks']
    if not receipts or any(not row.get('verified') for row in receipts):
        raise ValueError('predecessor_source_integrity_failed')
    prepared = read(protocol / 'prepared_arms.json')
    template = prepared['skills_full_shared_recovery']['launch_command_pending_confirmation']
    commands = {}
    reused = {}
    for arm, settings in ARMS.items():
        if settings['reuse']:
            directory = predecessor / settings['reuse']
            result = read(directory / 'exit_status.json')
            if result.get('exit_code') != 0 or result.get('attempted') != 10:
                raise ValueError('cannot_reuse_incomplete_arm:' + arm)
            skills = read(directory / 'skills_protocol.json')
            if skills['skills'] != settings['skills'] or (settings['skills'] == 'on' and
                (skills['level'] != settings['level'] or skills['candidate_expansion'] != settings['expansion'])):
                raise ValueError('reused_arm_settings_mismatch:' + arm)
            runtime = read(directory / 'modelpool_budget_protocol.json')
            if runtime['transport_limit'] != 4 or not runtime['model_source_integrity']['verified']:
                raise ValueError('reused_arm_budget_or_source_mismatch:' + arm)
            files = [directory / 'results.json', directory / 'resolved_config.yaml', directory / 'skills_protocol.json', directory / 'modelpool_budget_protocol.json']
            for p in directory.rglob('attempt_result.json'):
                files += [p, p.with_name('execution_trace.jsonl'), p.with_name(p.parent.name + '.mp4')]
            if len(list(directory.rglob('attempt_result.json'))) != 10:
                raise ValueError('reused_arm_missing_attempts:' + arm)
            reused[arm] = dict(directory=str(directory), complete_group_reuse=True,
                files_sha256={str(p.relative_to(root)): digest(p) for p in files})
        else:
            commands[arm] = missing_command(template, arm, future_output / 'manifest.json', future_output / arm)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'development10.json').write_bytes((protocol / 'development10.json').read_bytes())
    metadata = dict(status='prepared_not_started', planned_four_arm_attempts=40,
        completed_reused_attempts=20, new_planned_attempts=20, manifest_indices=expected['original_scenario_indices'],
        manifest_sha256=digest(protocol / 'development10.json'), original_protocol=str(protocol),
        model_pool_transition=pool_transition,
        reused_complete_arms=reused, missing_arm_commands=commands, source_sha256=frozen,
        controls=COMMON, max_screening_queries=40, actual_future_output=str(future_output),
        comparisons=dict(knowledge='unified_catalog_no_knowledge vs skills_no_expansion: same catalog expansion OFF',
                         expansion='skills_no_expansion vs skills_with_expansion: same full skill level'),
        interpretation=['base vs structured changes the closed-catalog interface, not only a single validation function',
            'structured arms share compact view, scheduling and both preflights; only documents/expansion vary',
            'reuse is complete R3 groups, never truncated prefixes; this is development, not unseen testing',
            'runtime must recheck model source transitions, scene/source hashes, disk and entirely idle GPUs',
            'model-pool fallback changes actual model mix and must be disclosed',
            'no automatic start while holdout validation is queued or active'], API_calls=0, new_executed_frames=0)
    (destination / 'prepared_four_arm_plan.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home2/zhaoyx/scenario-dreamer'))
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--predecessor', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--future-output', type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.root, args.protocol, args.predecessor, args.destination, args.future_output)
    print(json.dumps(dict(status=plan['status'], reused=20, new_planned=20, API_calls=0), ensure_ascii=False))


if __name__ == '__main__':
    main()
