"""独立v2开发层：澄清证据语义、压缩显示，不修改v1候选与执行器。"""
import copy
import json
import math
import numpy as np
from policies.skills_planner import SkillsAttackPlanner
from policies.intent_validation import canonical_hash


def round_display(value):
    """只用于模型可见视图；全精度目录继续负责预检和编译。"""
    if isinstance(value,dict):return {k:round_display(v) for k,v in value.items()}
    if isinstance(value,list):return [round_display(v) for v in value]
    if isinstance(value,float):return round(value,3)
    return value


def compact_prompt(prompt):
    prefix,text=prompt.split('\n',1)
    payload=json.loads(text);context=payload['context'];view=copy.deepcopy(context)
    # 完整拒绝审计留在trace中；模型只能选择已发布目录，无需重复阅读失败目录。
    view.pop('semantic_precheck',None)
    view.pop('candidate_rejections',None)
    view.pop('candidate_audit',None)
    for candidate in view['candidates']:
        candidate.pop('screening_model',None)
        candidate.pop('scene_conditions',None)
        candidate.pop('avoidability',None)
        if not context.get('profile_enabled'):
            for name in ('profile_match','profile_evidence_frames','vulnerability'):
                candidate.pop(name,None)
    view=round_display(view)
    payload['context']=view
    payload['display_contract']=dict(precision_decimals=3,geometry_for_compilation='full precision catalog in local trace',
        catalog_hash_basis='full precision source catalog; copy the supplied identity, not the display hash',
        display_hash=canonical_hash(view),candidate_ids_unchanged=True)
    prefix+=(' Rank a best-first prefix of at most SIX candidates; do not return the entire catalog. '
             'scene_analysis should be at most 80 words and each candidate reason at most 35 words. '
             'Use supplied numeric evidence, distinguish measured facts and extrapolation, and never call a background target the ego. '
             'Do not add display_hash to the response schema.')
    return prefix+'\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False)


class CompactSkillsPlanner(SkillsAttackPlanner):
    def build_context(self,env,state,planner,profile_context=None):
        context=super().build_context(env,state,planner,profile_context)
        ego=np.asarray(state['ego_state'],float)
        forward=np.array([math.cos(ego[4]),math.sin(ego[4])])
        right=np.array([forward[1],-forward[0]])
        features=[]
        for actor in state['agents']:
            s=np.asarray(actor['state'],float);relative=s[:2]-ego[:2];velocity=s[2:4]-ego[2:4]
            features.append(dict(target_id=actor['id'],role='background_actor_not_ego',
                relative_right_m=float(relative@right),relative_forward_m=float(relative@forward),
                relative_velocity_right_mps=float(velocity@right),relative_velocity_forward_mps=float(velocity@forward),
                heading_difference_rad=float(math.atan2(math.sin(s[4]-ego[4]),math.cos(s[4]-ego[4])))))
        count=int(np.count_nonzero(env.agent_active));statics=len(env.get_static_obstacles())
        for c in context['candidates']:
            margin=c.get('background_safety_margin_m')
            status='no_non_target_objects' if count==1 and statics==0 else ('sampled_margin' if margin is not None else 'measurement_unavailable')
            c['background_safety_assessment']=dict(status=status,non_target_dynamic_count=max(0,count-1),
                static_primitive_count=statics,threshold_m=self.builder.margin,minimum_sampled_margin_m=margin,
                final_joint_safety_required=True)
        context.update(scene_features=features,ego_role='separate ego; candidate target_id never refers to ego',
            evidence_notes=dict(initial_velocity_delta='difference between first 1s mean segment velocity and current target velocity; not a risk reward',
                estimated_ttc='CV sampled contact in first3s; values above3s extrapolate terminal gap and closing speed; not measured event time',
                family='code intent label; merge_conflict is not proof of real lane topology',
                profile='absent or low sample count is uncertainty; do not invent a driving style'))
        return context

    def plan(self,planner,*args,**kwargs):
        original=planner._request_attack_plan
        def request(prompt,scene_image=None):
            rendered=compact_prompt(prompt)
            decoded=json.loads(rendered.split('\n',1)[1])
            planner.last_trace.setdefault('transport_views',[]).append(decoded['display_contract'])
            return original(rendered,scene_image=scene_image)
        planner._request_attack_plan=request
        try:return super().plan(planner,*args,**kwargs)
        finally:planner._request_attack_plan=original
