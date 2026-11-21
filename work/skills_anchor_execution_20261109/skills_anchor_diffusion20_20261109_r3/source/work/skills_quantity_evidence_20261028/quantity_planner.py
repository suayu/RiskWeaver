"""默认关闭的事实补充；原schema/验证/预算继续生效，不增加返回字段或门控。"""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/skills_incremental_evidence_20261025'))
from incremental_planner import IncrementalSkillsPlanner
from quantity_facts import build_quantity_facts


def skill_body():
    raw=(HERE/'SKILL.md').read_text(encoding='utf-8')
    return raw.split('---',2)[-1].strip()


def render_quantity_prompt(prompt):
    prefix,data=prompt.rsplit('\n',1);payload=json.loads(data)
    if 'quantity_facts' not in payload['context']:raise ValueError('quantity_facts_missing_from_actual_prompt')
    # 只增加知识和输入，不修改既有输出契约，也不替模型改写原始回复。
    return prefix+'\n'+skill_body()+'\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False)


class QuantityEvidencePlanner(IncrementalSkillsPlanner):
    def __init__(self,sim_cfg,call_budget=4,extra_recipes=False,quantity_evidence=False):
        if not isinstance(quantity_evidence,bool):raise ValueError('quantity_evidence_requires_boolean')
        super().__init__(sim_cfg,call_budget,extra_recipes)
        self.quantity_evidence=quantity_evidence

    def build_context(self,env,state,planner,profile_context=None):
        context=super().build_context(env,state,planner,profile_context)
        if self.quantity_evidence:context['quantity_facts']=build_quantity_facts(state,context,int(env.current_step))
        return context

    def plan(self,planner,*args,**kwargs):
        if not self.quantity_evidence:return super().plan(planner,*args,**kwargs)
        request=planner._request_attack_plan
        planner._request_attack_plan=lambda prompt,scene_image=None:request(render_quantity_prompt(prompt),scene_image=scene_image)
        try:return super().plan(planner,*args,**kwargs)
        finally:
            planner._request_attack_plan=request
            planner.last_trace['quantity_evidence_protocol']=dict(version='quantity_evidence_offline_prototype_20261028_v1',enabled=True,
                knowledge_sha256=hashlib.sha256((HERE/'SKILL.md').read_bytes()).hexdigest(),
                response_schema_unchanged=True,extra_requests_outside_existing_budget=0,
                new_acceptance_gates=0,candidate_order_unchanged=True,
                prose_and_tactics_and_actual_threat_certified=False,continuous_refinement_and_safety_changed=False)
