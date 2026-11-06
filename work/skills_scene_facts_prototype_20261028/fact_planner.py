"""P1事实约束原型，默认关闭；未装入当前252源码冻结实验。"""
import copy
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/skills_incremental_evidence_20261025'))
from incremental_planner import IncrementalSkillsPlanner
from scene_facts import build_cards, validate_claims, FIELDS


def render_fact_prompt(prompt):
    prefix, content=prompt.rsplit('\n',1)
    payload=json.loads(content)
    if 'scene_fact_cards' not in payload['context']:
        raise ValueError('scene_fact_cards_not_in_real_prompt')
    old='interaction_basis, contrast_direction}'
    new='interaction_basis, contrast_direction, scene_fact_id, source_step, longitudinal_relation, lateral_relation, heading_relation}'
    if prefix.count(old)!=1:
        raise ValueError('scene_fact_contract_prefix_drift')
    prefix=prefix.replace(old,new)
    prefix+=('\nFor EVERY ranked candidate, copy scene_fact_id from scene_fact_cards[candidate].fact_id and copy '
        'source_step, longitudinal_relation, lateral_relation, heading_relation from that same card. '
        'These describe current pose and kinematic heading, not map topology. '
        'crossing_headings must not be described as oncoming/opposite; behind must not be described as ahead. '
        'When aligned, do not invent a left/right or front/back label. '
        'Keep the original reason and scene_analysis consistent with these facts; do not claim this checker certifies prose or tactics. '
        'A correct copy is identity/consistency evidence, not proof of actual ego threat. '
        'Preserve the catalog ranking decision, active-window evidence, uncertainty, and the existing request budget.\n')
    return prefix+json.dumps(payload,ensure_ascii=False,allow_nan=False)


class SceneFactPlanner(IncrementalSkillsPlanner):
    def __init__(self, sim_cfg, call_budget=4, extra_recipes=False, fact_grounding=False):
        if not isinstance(fact_grounding,bool):
            raise ValueError('scene_fact_requires_explicit_boolean')
        super().__init__(sim_cfg,call_budget,extra_recipes)
        self.fact_grounding=fact_grounding
        self.scene_fact_cards={}

    def build_context(self,env,state,planner,profile_context=None):
        context=super().build_context(env,state,planner,profile_context)
        if self.fact_grounding:
            self.scene_fact_cards=build_cards(state,context['candidates'],int(env.current_step))
            context['scene_fact_cards']=self.scene_fact_cards
            context['scene_fact_validation_scope']='typed_source_claims_only; raw_prose_not_certified; no_new_safety_or_distance_gate'
        return context

    def validate_response(self,response,request_id,state_hash,catalog_hash,catalog,obstacles):
        if not self.fact_grounding:
            return super().validate_response(response,request_id,state_hash,catalog_hash,catalog,obstacles)
        validate_claims(response,self.scene_fact_cards)
        original=copy.deepcopy(response)
        for row in original['grounding'].values():
            for key in FIELDS:
                row.pop(key)
        # 对象/有效期/引用/contrast/schema等原验证必须继续执行，不能只验新增字段。
        return super().validate_response(original,request_id,state_hash,catalog_hash,catalog,obstacles)

    def plan(self,planner,*args,**kwargs):
        if not self.fact_grounding:
            return super().plan(planner,*args,**kwargs)
        request=planner._request_attack_plan
        planner._request_attack_plan=lambda prompt,scene_image=None:request(render_fact_prompt(prompt),scene_image=scene_image)
        try:
            return super().plan(planner,*args,**kwargs)
        finally:
            planner._request_attack_plan=request
            planner.last_trace['scene_fact_protocol']=dict(version='typed_scene_facts_prototype_20261028',
                enabled=True,typed_validation='accepted' if planner.last_trace.get('plan_validation')=='accepted' else 'not_established',
                checks=['source_pose_labels','kinematic_heading_relation','source_time','fact_identity'],
                raw_prose_certified=False,tactical_correctness_certified=False,extra_requests_outside_existing_budget=0,
                continuous_refinement_or_safety_changes=False)
