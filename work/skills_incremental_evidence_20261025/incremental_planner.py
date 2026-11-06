"""增量证据的独立高级规划原型；不替换已冻结v5，新增配方默认关闭。"""
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'work/skills_priority_v5_20261024')]
from priority_planner import PrioritySkillsPlanner, PREFIX
from policies.skill_registry import SkillRegistry
from policies.intent_validation import validate_world_intent
from incremental_evidence import incremental_card
from active_window_recipes import screened_proposals


def render_incremental_prompt(prompt):
    """只压缩模型视图；全精度事实和原始回答仍进入trace，编译规则不变。"""
    # 上游v5前缀有多行，JSON仍由单行dumps生成；不能从第一行截取。
    _, payload = prompt.rsplit('\n', 1)
    payload = json.loads(payload)
    context = payload['context']
    cards = context.pop('incremental_cards')
    context.pop('incremental_recipe_audit', None)
    # 只显示计算所需的对比事实，不反复传输安全说明和同一语义指标。
    for cid, card in cards.items():
        facts = context['priority_evidence'][cid]['facts']
        safety = facts[cid+':safety']
        facts[cid+':safety'] = dict(supplied_margin_m=safety.get('supplied_margin_m'),
             positive_ego_area_certified=False, final_joint_validation_required=True)
        facts[cid+':contrast'] = {k:card[k] for k in [
            'candidate_min_gap_m', 'candidate_min_time_s', 'natural_CV_min_gap_m',
            'natural_CV_min_time_s', 'min_gap_difference_m',
            'same_time_difference_at_candidate_min_m', 'proxy_direction']}
    prefix = PREFIX.replace(
        '{target_id, target_role, active_horizon_seconds, evidence_ids, interaction_basis}',
        '{target_id, target_role, active_horizon_seconds, evidence_ids, interaction_basis, contrast_direction}')
    prefix += ('\nFor EVERY ranked ID also cite its :contrast fact and copy contrast_direction from its proxy_direction. '
        'Current distance closing alone is not evidence that the attack added threat: compare with natural target CV. '
        'Prefer supported active-window incremental interaction; explain tradeoffs rather than saying "strongest" without a criterion. '
        'Negative/no difference is not proof that an intent is impossible: a later-forecast or exploratory probe remains allowed, '
        'but label it honestly. All distance differences are proxies, not measured risk/ego response or a safety certificate. '
        'Use qualitative reasons with fact IDs instead of recopying many numbers. Do not request more calls or change duration/solver.\n')
    return prefix+json.dumps(payload, ensure_ascii=False, allow_nan=False)


class IncrementalSkillsPlanner(PrioritySkillsPlanner):
    def __init__(self, sim_cfg, call_budget=4, extra_recipes=False):
        if not isinstance(extra_recipes, bool):
            raise ValueError('explicit_boolean_recipe_switch_required')
        super().__init__(sim_cfg, call_budget)
        if extra_recipes and not self.expansion:
            raise ValueError('active_window_recipes_require_candidate_expansion')
        self.registry = SkillRegistry(HERE/'skills', getattr(self.settings, 'level', 'full'))
        self.extra_recipes = extra_recipes
        self.recipe_audit = []
        self.incremental_cards = {}

    def expanded_candidates(self, env, state, planner):
        existing = super().expanded_candidates(env, state, planner)
        if not self.extra_recipes:
            return existing
        additions, self.recipe_audit = screened_proposals(env, state, self.builder,
            lambda snapshot, candidate:validate_world_intent(planner, snapshot, candidate))
        return existing+additions

    def build_context(self, env, state, planner, profile_context=None):
        self.recipe_audit = []
        context = super().build_context(env, state, planner, profile_context)
        self.incremental_cards = {c['candidate_id']:incremental_card(state, c, self.dt, self.interval)
                                  for c in context['candidates']}
        for cid, card in self.incremental_cards.items():
            self.priority_evidence[cid]['facts'][cid+':contrast'] = card
        context.update(incremental_cards=self.incremental_cards,
            incremental_recipe_switch=self.extra_recipes, incremental_recipe_audit=self.recipe_audit,
            incremental_scope='motion proxy vs constant velocity; not real-policy causal effect; no new gate')
        return context

    def validate_response(self, response, request_id, state_hash, catalog_hash, catalog, obstacles):
        if not isinstance(response, dict) or not isinstance(response.get('grounding'), dict):
            raise ValueError('incremental_grounding_missing')
        legacy = copy.deepcopy(response)
        for cid, row in legacy['grounding'].items():
            if not isinstance(row, dict) or 'contrast_direction' not in row:
                raise ValueError('incremental_direction_missing')
            direction = row.pop('contrast_direction')
            if cid not in self.incremental_cards or direction != self.incremental_cards[cid]['proxy_direction']:
                raise ValueError('incremental_direction_contradicts_fact')
            if cid+':contrast' not in row.get('evidence_ids', []):
                raise ValueError('incremental_reference_missing')
        super().validate_response(legacy, request_id, state_hash, catalog_hash, catalog, obstacles)

    def plan(self, planner, *args, **kwargs):
        request = planner._request_attack_plan
        planner._request_attack_plan = lambda prompt, scene_image=None:request(
            render_incremental_prompt(prompt), scene_image=scene_image)
        try:
            return super().plan(planner, *args, **kwargs)
        finally:
            planner._request_attack_plan = request
            planner.last_trace['incremental_protocol'] = dict(version='incremental-prototype-20261025',
                extra_recipes=self.extra_recipes, continuous_refinement_changed=False,
                direction_validation='accepted' if planner.last_trace.get('plan_validation') == 'accepted' else 'not_established',
                prose_accuracy_and_actual_threat_not_certified=True)
