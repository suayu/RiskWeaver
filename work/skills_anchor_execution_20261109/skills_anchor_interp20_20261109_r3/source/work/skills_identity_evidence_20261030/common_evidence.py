"""共享的身份复制参考与数值比较事实；不代填响应、不改变目录或接受门槛。"""
from pathlib import Path
import copy,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent


def comparison_facts(context):
    rows={}
    for candidate in context['candidates']:
        cid=candidate['candidate_id'];contrast=context['incremental_cards'][cid]
        rows[cid]=dict(target_id=candidate['target_id'],active_window_min_center_gap_m=contrast['candidate_min_gap_m'],
            natural_CV_window_min_center_gap_m=contrast['natural_CV_min_gap_m'],
            natural_min_minus_candidate_min_m=contrast['min_gap_difference_m'],
            natural_gap_at_candidate_min_time_minus_candidate_min_m=contrast['same_time_difference_at_candidate_min_m'],
            proxy_direction=contrast['proxy_direction'],background_margin_m=candidate.get('background_safety_margin_m'),
            event_time_seconds=candidate.get('event_time_seconds'),event_within_current_lease=candidate.get('event_within_current_lease'))
    def extreme(field,minimum):
        finite={cid:r[field] for cid,r in rows.items() if r[field] is not None and np.isfinite(r[field])}
        if not finite:return dict(value=None,candidate_ids=[],reason='no_finite_supplied_values')
        value=(min if minimum else max)(finite.values())
        return dict(value=value,candidate_ids=[cid for cid,v in finite.items() if abs(v-value)<=1e-6],numeric_tolerance=1e-6)
    return dict(per_candidate=rows,smallest_active_gap=extreme('active_window_min_center_gap_m',True),
        largest_added_closeness=extreme('natural_min_minus_candidate_min_m',False),
        largest_finite_background_margin=extreme('background_margin_m',False),smallest_finite_background_margin=extreme('background_margin_m',True),
        scope='source_hypothesis_center_distances_only; no_execution_or_threat_or_safety_certificate',
        different_time_minima_and_same_time_comparisons_are_different_quantities=True,
        candidate_order_and_original_validators_unchanged=True)


def add_facts(context):
    output=copy.deepcopy(context);output['comparison_facts']=comparison_facts(output);return output


def render_prompt(prompt):
    prefix,raw=prompt.rsplit('\n',1);payload=json.loads(raw)
    identities={k:payload[k] for k in ['request_id','state_hash','catalog_hash']}
    if 'comparison_facts' not in payload['context']:raise ValueError('comparison_facts_missing_before_catalog_hash')
    feedback=payload.get('repair_feedback')
    if feedback and feedback.get('reason')=='skills_identity_mismatch':feedback['expected_identity']=copy.deepcopy(identities)
    # 末尾重复原身份，只帮助模型准确复制；响应仍须原完整64位哈希逐字符匹配。
    payload['required_identity_copy_exactly']=identities
    skill=(HERE/'SKILL.md').read_text(encoding='utf-8').split('---',2)[-1].strip()
    return prefix+'\n'+skill+'\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False)


def install_common_evidence(engine):
    build=engine.build_context;plan=engine.plan
    def build_context(*args,**kwargs):return add_facts(build(*args,**kwargs))
    def controlled(planner,*args,**kwargs):
        request=planner._request_attack_plan
        planner._request_attack_plan=lambda prompt,scene_image=None:request(render_prompt(prompt),scene_image=scene_image)
        try:return plan(planner,*args,**kwargs)
        finally:
            planner._request_attack_plan=request
            planner.last_trace['common_identity_evidence']=dict(version='identity_tail_and_exact_comparisons_20261030_v1',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),skill_sha256=hashlib.sha256((HERE/'SKILL.md').read_bytes()).hexdigest(),
                response_identity_not_rewritten=True,original_validators_and_budget_preserved=True,
                supplied_facts_do_not_certify_prose_or_tactical_optimality=True)
    engine.build_context=build_context;engine.plan=controlled
    return engine
