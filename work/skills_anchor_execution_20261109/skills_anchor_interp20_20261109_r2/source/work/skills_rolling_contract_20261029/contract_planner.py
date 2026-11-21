"""独立契约修订：明确对象映射并记录几何等价候选，不宽松接收错误响应。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/skills_rolling_intent_20261028'))
from rolling_planner import RollingIntentPlanner

CONTRACT='''
STRICT JSON TYPES (same existing response fields; do not add fields):
candidate_reasons MUST be a JSON OBJECT, keyed by EVERY ranked candidate ID.
Each value MUST be a nonempty string. It MUST NOT be an array/list, even when
its length equals ranking. Example shape only, replace X with a supplied ID:
"ranking":["X"], "candidate_reasons":{"X":"Specific reason using supplied facts"}.
grounding is also an OBJECT keyed by EVERY ranked ID, with the original required
typed evidence fields. Keep identity strings exactly as supplied. Empty ranking
requires candidate_reasons={} and grounding={}; no_attack_reason must explain why.
The example is a shape, not permission to invent X or extra keys. If repairing
skills_missing_candidate_reasons, replace the reasons ARRAY with an ID-keyed OBJECT;
do not merely rewrite array strings. No response is accepted by post-hoc conversion.
'''


def deduplicate_intents(candidates,tolerance_m=1e-6):
    """只合并同目标/策略/有效期、所有锚点欧氏距离均小于数值容差的候选。"""
    if not np.isfinite(tolerance_m) or tolerance_m<=0:raise ValueError('invalid_numeric_equivalence_tolerance')
    kept=[];audit=[]
    for candidate in candidates:
        a=np.asarray(candidate['anchors'],float)
        if a.shape!=(4,2) or not np.isfinite(a).all():
            kept.append(copy.deepcopy(candidate));continue  # 非法输入仍由既有验证器处理，不能修成可接受。
        representative=None;deviation=None
        for prior in kept:
            if any(candidate.get(k)!=prior.get(k) for k in ['target_id','strategy','duration']):continue
            b=np.asarray(prior['anchors'],float)
            if b.shape!=(4,2) or not np.isfinite(b).all():continue
            value=float(np.linalg.norm(a-b,axis=1).max())
            if value<=tolerance_m:representative=prior['candidate_id'];deviation=value;break
        if representative is None:kept.append(copy.deepcopy(candidate))
        else:audit.append(dict(removed_candidate_id=candidate['candidate_id'],representative_id=representative,
            maximum_anchor_deviation_m=deviation,numeric_equivalence_tolerance_m=tolerance_m,
            unchanged_kept_coordinates=True,not_a_safety_threshold=True))
    return kept,audit


def render_contract_prompt(prompt):
    """增加可执行的类型说明和原始错误反馈，保持原响应schema与API调用次数。"""
    prefix,raw=prompt.rsplit('\n',1);payload=json.loads(raw)
    feedback=payload.get('repair_feedback')
    if feedback and feedback.get('reason')=='skills_missing_candidate_reasons':
        prior=feedback.get('previous_response') or {};ranking=prior.get('ranking') or []
        feedback['expected_candidate_reasons']=dict(type='object',keys=ranking,value_type='nonempty_string',
            lists_forbidden=True,example_shape={'<a_ranked_candidate_id>':'<specific_supported_reason>'})
    # 大体积去重审计保留完整trace，不把删除的ID再当作当前目录交给模型。
    payload['context'].pop('numeric_intent_deduplication',None)
    return prefix+'\n'+CONTRACT+'\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False)


class ContractRollingPlanner(RollingIntentPlanner):
    def __init__(self,sim_cfg,call_budget=4,extra_recipes=False,rolling_intent=False,explicit_contract=False,deduplicate=False):
        if not isinstance(explicit_contract,bool) or not isinstance(deduplicate,bool):raise ValueError('explicit_boolean_contract_and_dedup_required')
        super().__init__(sim_cfg,call_budget,extra_recipes,rolling_intent)
        if deduplicate and not rolling_intent:raise ValueError('dedup_requires_rolling_catalog')
        self.explicit_contract=explicit_contract;self.deduplicate=deduplicate;self.dedup_audit=[]

    def expanded_candidates(self,env,state,planner):
        candidates=super().expanded_candidates(env,state,planner)
        if not self.deduplicate:return candidates
        result,self.dedup_audit=deduplicate_intents(candidates)
        return result

    def build_context(self,env,state,planner,profile_context=None):
        context=super().build_context(env,state,planner,profile_context)
        if self.deduplicate:context['numeric_intent_deduplication']=dict(removed=self.dedup_audit,
            tolerance_m=1e-6,scope='numeric_equivalence_only_not_new_safety_or_feasibility_gate')
        return context

    def plan(self,planner,*args,**kwargs):
        request=planner._request_attack_plan
        if self.explicit_contract:
            planner._request_attack_plan=lambda prompt,scene_image=None:request(render_contract_prompt(prompt),scene_image=scene_image)
        try:return super().plan(planner,*args,**kwargs)
        finally:
            planner._request_attack_plan=request
            if self.explicit_contract or self.deduplicate:
                planner.last_trace['explicit_contract_protocol']=dict(version='rolling_contract_v1_20261029',
                    explicit_contract=self.explicit_contract,numeric_deduplication=self.deduplicate,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    original_response_validator_preserved=True,response_arrays_not_silently_converted=True,
                    no_extra_requests=True,safety_and_diffusion_changed=False,closed_loop_gain_unverified=True)
