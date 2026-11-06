"""将已验证的后备物理15米口径接入静态请求与最终复查，默认关闭。"""
import copy
import hashlib
import json
from pathlib import Path

from .adapter import install

RULE_PATH = 'prototype/rw-static-route-physical15/SKILL.md'
VERSION = 'static-route-physical15-v1-20261102'
MARGIN = 15.0


def make_static(sim_cfg, call_budget=4, enabled=False):
    from policies.hybrid_skills_planner import StaticCombinationPlanner
    from policies.ego_profile import clean_json
    if not isinstance(enabled, bool):
        raise ValueError('static_physical15_flag_requires_bool')
    if not enabled:
        return StaticCombinationPlanner(sim_cfg, call_budget)

    class Physical15StaticPlanner(StaticCombinationPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            raw = Path(__file__).with_name('SKILL.md').read_bytes()
            self.physical15_skill_sha256 = hashlib.sha256(raw).hexdigest()
            manifest = json.loads(Path(__file__).with_name('manifest.json').read_text(encoding='utf-8'))
            if self.physical15_skill_sha256 != manifest['files']['SKILL.md']:
                raise ValueError('physical15_skill_manifest_hash_mismatch')
            load = self.registry.load
            def with_rule(*args, **kwargs):
                knowledge = copy.deepcopy(load(*args, **kwargs))
                if self.registry.level in ('analysis', 'full'):
                    knowledge['documents'].append(dict(path=RULE_PATH,
                        sha256=self.physical15_skill_sha256, text=raw.decode('utf-8')))
                    knowledge['version'] += '+' + VERSION
                return knowledge
            self.registry.load = with_rule
            self._physical15_receipt = None
            self._physical15_final_checks = []

        def build_context(self, env, state, planner, profile_context=None):
            # 独立预览目录也必须遵守同一规则；plan内已经有scope，不能再次嵌套改写AST。
            build = super().build_context
            if self._physical15_receipt is None:
                with install(planner, env, enabled=True, initial_margin=MARGIN):
                    context = build(env, state, planner, profile_context)
            else:
                context = build(env, state, planner, profile_context)
            context['static_clearance_policy'] = dict(version=VERSION,
                original_forward_minimum_m=15., route_footprint_entry_minimum_m=15.,
                fallback_initial_ego_OBB_margin_m=MARGIN,
                background_margin_m=self.builder.margin,
                original_success_path_preserved=True,
                applies_physical15_to_new_fallback_only=True,
                not_execution_threat_or_appearance_certificate=True)
            return context

        def select_obstacle(self, response, context, ranking):
            before = len(self._physical15_receipt['events'])
            check = dict(obstacle_candidate_id=response.get('obstacle_candidate_id'),
                         fallback_initial_ego_OBB_margin_m=MARGIN)
            try:
                result = super().select_obstacle(response, context, ranking)
            except ValueError as error:
                check.update(accepted=False, error_type=type(error).__name__, reason=str(error))
                raise
            else:
                check['accepted'] = True
                return result
            finally:
                check['new_route_check_events'] = len(self._physical15_receipt['events']) - before
                self._physical15_final_checks.append(check)

        def plan(self, planner, env, *args, **kwargs):
            if self._physical15_receipt is not None:
                raise RuntimeError('static_physical15_plan_reentry_not_supported')
            with install(planner, env, enabled=True, initial_margin=MARGIN) as receipt:
                self._physical15_receipt = receipt
                self._physical15_final_checks = []
                try:
                    return super().plan(planner, env, *args, **kwargs)
                finally:
                    # 审计值也可能包含NumPy标量；只做JSON类型适配，不改验证结果。
                    try:
                        if isinstance(getattr(planner, 'last_trace', None), dict):
                            planner.last_trace['static_physical15'] = clean_json(dict(
                                version=VERSION, geometry=copy.deepcopy(receipt),
                                final_checks=copy.deepcopy(self._physical15_final_checks),
                                skill_sha256=self.physical15_skill_sha256,
                                background_margin_m=self.builder.margin,
                                actual_LLM_or_execution_success_not_certified=True))
                    finally:
                        self._physical15_receipt = None

    return Physical15StaticPlanner(sim_cfg, call_budget)
