"""让旧自由规划与Skills共享同一可选边界和最终静态复核；仅评估，不注入。"""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path

from work.skills_static_physical15_20261102.adapter import install as route_scope

HERE = Path(__file__).resolve().parent
VERSION = 'common-highlevel-static-gate-v1-20261102'


def load_rule():
    """实验规则固定并校验版本；关闭开关时不加载。"""
    raw = (HERE/'COMMON_STATIC_RULE.md').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    if digest != manifest['files']['COMMON_STATIC_RULE.md']:
        raise ValueError('common_static_rule_manifest_mismatch')
    return raw.decode('utf-8'), digest


def state_hash(state):
    from policies.ego_profile import clean_json
    from policies.intent_validation import canonical_hash
    return canonical_hash(clean_json(state))


@contextmanager
def common_scope(planner, env, enabled=False):
    """两条链共用一个scope；最终复核也在同一scope内，结束恢复实例方法。"""
    if not isinstance(enabled, bool):
        raise ValueError('common_static_gate_requires_boolean')
    receipt = dict(enabled=enabled, version=VERSION, validation_calls=[])
    if not enabled:
        yield receipt
        return
    if not getattr(planner, '_use_local_coordinate', True):
        raise ValueError('common_static_gate_requires_original_local_coordinate_contract')
    if getattr(planner, '_rw_common_static_gate_active', False):
        raise RuntimeError('nested_common_static_gate_not_supported')
    marker_present = '_rw_common_static_gate_active' in planner.__dict__
    marker = planner.__dict__.get('_rw_common_static_gate_active')
    planner._rw_common_static_gate_active = True
    try:
        with route_scope(planner, env, enabled=True, initial_margin=15.) as geometry:
            original = planner._validate_obstacle_plan
            def whole_combination(payload, normalized):
                checked = original(payload, normalized)
                intact = isinstance(payload, list) and len(checked) == len(payload)
                receipt['validation_calls'].append(dict(
                    submitted=copy.deepcopy(payload), accepted_before_whole_group_check=copy.deepcopy(checked),
                    whole_group_preserved=intact))
                # 原调用可继续保留有效动态部分；不把部分静态组当完整组合通过。
                return checked if intact else []
            planner._validate_obstacle_plan = whole_combination
            receipt['geometry'] = geometry
            try:
                yield receipt
            finally:
                planner._validate_obstacle_plan = original
    finally:
        if marker_present:
            planner._rw_common_static_gate_active = marker
        else:
            del planner._rw_common_static_gate_active


def plan_once(planner, env, state, instruction, *, skills_engine=None,
              builder=None, enabled=False, profile_context=None, scene_image=None):
    """同一评估调用支持原generate_attack_plan或现有Skills.plan；不请求额外修复。"""
    if not isinstance(enabled, bool):
        raise ValueError('common_static_gate_requires_boolean')
    input_digest = state_hash(state)
    selected_instruction = instruction
    digest = None
    if enabled:
        if builder is None:
            raise ValueError('explicit_common_static_builder_required')
        if input_digest != state_hash(env.get_state_for_planning()):
            raise ValueError('planning_input_differs_from_current_environment')
        rule, digest = load_rule()
        selected_instruction = ([instruction] if isinstance(instruction, str) else list(instruction)) + [rule]
    with common_scope(planner, env, enabled) as receipt:
        if skills_engine is None:
            returned = planner.generate_attack_plan(state, selected_instruction, scene_image=scene_image)
        else:
            returned = skills_engine.plan(planner, env, state, selected_instruction,
                profile_context=profile_context, scene_image=scene_image)
        # 原输出单独保留；评估输出不能回写模型回复或原实验。
        result = dict(version=VERSION, enabled=enabled, path='legacy_off' if skills_engine is None else 'skills',
            returned_plan=copy.deepcopy(returned), evaluated_plan=copy.deepcopy(returned),
            common_rule_sha256=digest, scope=receipt, source_state_sha256=input_digest,
            actual_execution_frames=0, D=None, collision=None,
            fresh_LLM_and_semantic_correctness_not_certified=True)
        if not enabled:
            result['final_common_static_validation'] = None
            return result
        if getattr(planner, 'last_request_failed', False):
            result.update(evaluated_plan=None, status='planner_service_failure',
                          final_common_static_validation=None)
            return result
        if input_digest != state_hash(state) or input_digest != state_hash(env.get_state_for_planning()):
            result.update(evaluated_plan=None, status='stale_planning_state',
                          final_common_static_validation=None)
            return result
        result['status'] = 'returned_without_service_failure'
        placements = returned.get('obstacle_plan') if isinstance(returned, dict) else None
        if placements:
            from policies.static_attack_catalog import validate_placements
            # 旧自由输出和Skills选择均进行同一真实整组/背景/面积复核，不做目录ID限制。
            check = validate_placements(env, planner, builder, placements)
            result['final_common_static_validation'] = check
            if not check['accepted']:
                result['evaluated_plan']['obstacle_plan'] = []
                result['static_rejection_reason'] = check['reason']
        else:
            result['final_common_static_validation'] = None
        from policies.ego_profile import clean_json
        return clean_json(result)
