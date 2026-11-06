"""优先场景理解与有依据的离散攻击；不修改任何连续细化/安全求解代码。"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'work/skills_v2_compact_dev_20261020'),
               str(ROOT/'work/skills_recovery_v4_20261023')]
from route_tail_preflight import RouteTailReadyPlanner
from policies.skill_registry import SkillRegistry
from evidence import candidate_evidence, validate_grounding
from bailian_transport import compatible_request


PREFIX = '''Rank only the supplied discrete attack catalog. Return one JSON object with exactly:
request_id, state_hash, catalog_hash, ranking, scene_analysis, candidate_reasons, no_attack_reason, grounding.
ranking is a best-first prefix of at most SIX candidate IDs. Each ID needs a concrete candidate_reasons string.
grounding maps EVERY ranked ID to exactly {target_id, target_role, active_horizon_seconds, evidence_ids, interaction_basis}.
target_role must be background_vehicle. Copy the numeric target and active horizon from priority_evidence.
evidence_ids must cite the candidate's :actor and :validity facts, plus :active when basis is active_window_proxy or exploratory_probe,
or :after when basis is later_forecast_only. Never invent evidence IDs. These are proxies, not certified collisions or measured D.
Priority1: correctly identify the scene, active background targets and usable intents; use a supported feasible probe when available.
Priority2: prefer supported ego interaction during the ACTUAL active window; do not count a later anchor as already executed.
An exploratory probe is allowed but must be described as uncertain. Do not rank acceleration itself as attack reward.
Priority3: obey applicable enabled natural-language constraints within the supplied catalog; never invent a matching candidate.
Priority4: use supplied difficulty evidence only when enabled; do not change u, solver parameters, metric definitions or safety checks.
Copy all three identity strings. Empty ranking needs a concrete no_attack_reason; otherwise no_attack_reason is empty and grounding keys match ranking.
Use at most80 words for scene_analysis and35 per reason. No trajectories, controls, obstacle fields or invented IDs.
'''


def render_priority_prompt(prompt):
    # 重新生成唯一响应契约，避免旧前缀的“exactly”字段集与新字段互相矛盾。
    _, content = prompt.split('\n', 1)
    payload = json.loads(content)
    if payload['context'].get('obstacle_candidates'):
        raise ValueError('priority_v5_dynamic_stage_only; static branch remains legacy')
    return PREFIX + '\n' + json.dumps(payload, ensure_ascii=False, allow_nan=False)


class PrioritySkillsPlanner(RouteTailReadyPlanner):
    def __init__(self, sim_cfg, call_budget=4):
        super().__init__(sim_cfg, call_budget)
        if getattr(sim_cfg, 'obstacles', None) is not None and getattr(sim_cfg.obstacles, 'enabled', False):
            raise ValueError('priority_v5_first_stage_requires_obstacles_off')
        self.registry = SkillRegistry(Path(__file__).with_name('skills'), getattr(self.settings, 'level', 'full'))
        self.dt = float(sim_cfg.dt)
        self.interval = float(sim_cfg.traffic_model.anchor_guidance.anchor_interval_seconds)
        self.priority_evidence = {}

    def build_context(self, env, state, planner, profile_context=None):
        context = super().build_context(env, state, planner, profile_context)
        self.priority_evidence = candidate_evidence(state, context['candidates'], self.dt, self.interval)
        context['priority_evidence'] = self.priority_evidence
        context['priority_order'] = ['scene_understanding_and_correct_discrete_attack',
            'actual_ego_threat', 'enabled_natural_language_control', 'enabled_target_difficulty']
        context['evidence_scope'] = 'actor/time/grounding verified; opportunity and threat still require independent execution audit'
        return context

    def validate_response(self, response, request_id, state_hash, catalog_hash, catalog, obstacles):
        if obstacles:
            raise ValueError('priority_v5_static_not_yet_validated')
        if not isinstance(response, dict) or 'grounding' not in response:
            raise ValueError('priority_missing_grounding')
        original = {k: v for k, v in response.items() if k != 'grounding'}
        super().validate_response(original, request_id, state_hash, catalog_hash, catalog, False)
        validate_grounding(response, catalog, self.priority_evidence)

    def plan(self, planner, *args, **kwargs):
        original_request = planner._request_attack_plan
        planner._request_attack_plan = lambda prompt, scene_image=None: original_request(
            render_priority_prompt(prompt), scene_image=scene_image)
        try:
            return super().plan(planner, *args, **kwargs)
        finally:
            planner._request_attack_plan = original_request
            planner.last_trace['priority_contract_version'] = 'v5-20261024'
            planner.last_trace['priority_validation_scope'] = 'identity/time/evidence-reference; not proof of tactical optimality or executed threat'
