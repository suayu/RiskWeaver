"""独立可关闭的高层时序候选；不更换diffusion、样本或联合安全验证。"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'work/skills_rolling_contract_20261029'),str(ROOT/'work/skills_quantity_evidence_20261028')]
from contract_planner import ContractRollingPlanner
from quantity_facts import motion_fact
from policies.intent_validation import validate_world_intent
from temporal_intents import fractional_minimum,guarded_fractional_validator,velocity_aligned_candidates

KNOWLEDGE='''
Use temporal_candidate_evidence. ego_right_m/right_m < 0 means LEFT, > 0 RIGHT;
forward_m < 0 means BEHIND, > 0 AHEAD. Use the explicit relation labels, not guesswork.
positive radial_gap_rate_mps means currently SEPARATING, not closing. Current source
motion and a candidate's future motion are different facts. A forecast is not ground truth.
The optional interaction check measures closest time ALONG the sparse piecewise-linear
intent, not just anchors at integer seconds. The radius remains 15m; this is not
background safety, dynamics validation or an executed threat certificate.
Continuation candidates reinstantiate the previous sparse acceleration signature from
CURRENT measured position AND velocity. Their fresh tail is a model extension,
not exact reuse of the old world path. Check current proxy benefit and continuity.
Do not claim a 'rolling' recipe preserves past intent when no prior episode exists.
Explain any needed revision using observed tracking/velocity differences and supplied
facts; do not silently call a missed old event successful. Prefer a supported continuation
when suitable, but do not force an unsafe or irrelevant prior intent. Preserve all JSON types.
'''


class TemporalCandidatePlanner(ContractRollingPlanner):
    def __init__(self,sim_cfg,call_budget=4,extra_recipes=False,rolling_intent=False,explicit_contract=False,deduplicate=False,
                 fractional_interaction=False,velocity_continuation=False,motion_evidence=False):
        flags=[fractional_interaction,velocity_continuation,motion_evidence]
        if any(not isinstance(f,bool) for f in flags):raise ValueError('temporal_flags_require_booleans')
        if any(flags) and not rolling_intent:raise ValueError('temporal_features_require_rolling_path')
        super().__init__(sim_cfg,call_budget,extra_recipes,rolling_intent,explicit_contract,deduplicate)
        self.fractional_interaction=fractional_interaction;self.velocity_continuation=velocity_continuation;self.motion_evidence=motion_evidence
        self.fractional_events=[];self.velocity_audit=[]

    def expanded_candidates(self,env,state,planner):
        existing=super().expanded_candidates(env,state,planner)
        if not self.velocity_continuation:return existing
        prior=self.previous_status.get('previous')
        if not prior:self.velocity_audit=[dict(available=False,reason='no_previous_accepted_intent')];return existing
        states=np.asarray(env.data_dict['agent'][-1]);target=prior['target_id']
        proposals,fit=velocity_aligned_candidates(prior,states[target],int(env.current_step),self.dt)
        self.velocity_audit=[dict(fit=fit)];lanes=[np.asarray(l)[:,:2] for l in env.scenario_dict.get('lanes',[]) if len(l)>1]
        result=[]
        for i,c in enumerate(proposals):
            c=dict(c,candidate_id=f'vc_{i}');estimates,reason=self.builder.screen(env,target,np.asarray(c['anchors']),lanes)
            audit=dict(candidate=c,screen_rejection=reason,accepted=False)
            if reason is None:
                semantic=validate_world_intent(planner,state,c);audit['semantic_validation']=semantic
                if semantic['accepted']:result.append(dict(c,**estimates,screened=True));audit['accepted']=True
            self.velocity_audit.append(audit)
        return existing+result

    def build_context(self,env,state,planner,profile_context=None):
        context=super().build_context(env,state,planner,profile_context)
        if not any([self.fractional_interaction,self.velocity_continuation,self.motion_evidence]):return context
        facts={}
        if self.motion_evidence:
            for actor in state['agents']:
                fact=motion_fact(state['ego_state'],actor['state'],actor['id'],int(env.current_step))
                fact.update(longitudinal_relation='ahead' if fact['forward_m']>.5 else 'behind' if fact['forward_m']<-.5 else 'aligned',
                    lateral_relation='right' if fact['right_m']>.5 else 'left' if fact['right_m']<-.5 else 'aligned',
                    relation_band_m=.5)
                facts[str(actor['id'])]=fact
        temporal={c['candidate_id']:fractional_minimum(c['anchors'],state['ego_state']) for c in context['candidates']}
        context['temporal_candidate_evidence']=dict(fractional_interaction_enabled=self.fractional_interaction,
            current_source_motion=facts,per_candidate_fractional_minimum=temporal,velocity_continuation_audit=self.velocity_audit,
            visible_continuation_candidate_ids=[c['candidate_id'] for c in context['candidates'] if c.get('recipe')=='current_state_velocity_aligned_prior_acceleration_v1'],
            not_ground_truth_future_or_executed_threat=True)
        return context

    def plan(self,planner,*args,**kwargs):
        if not any([self.fractional_interaction,self.velocity_continuation,self.motion_evidence]):return super().plan(planner,*args,**kwargs)
        request=planner._request_attack_plan;original=planner._validate_attack_plan
        had_own='_validate_attack_plan' in planner.__dict__;old_own=planner.__dict__.get('_validate_attack_plan')
        self.fractional_events=[];self.velocity_audit=[]
        if self.fractional_interaction:planner._validate_attack_plan=guarded_fractional_validator(original,self.fractional_events)
        def decorated(prompt,scene_image=None):
            prefix,raw=prompt.rsplit('\n',1);payload=json.loads(raw)
            # 提案原件保留trace，模型仅看最终目录和数值事实，不再读一大批已拒候选。
            payload['context']['temporal_candidate_evidence'].pop('velocity_continuation_audit',None)
            return request(prefix+'\n'+KNOWLEDGE+'\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False),scene_image=scene_image)
        planner._request_attack_plan=decorated
        try:return super().plan(planner,*args,**kwargs)
        finally:
            planner._request_attack_plan=request
            if self.fractional_interaction:
                if had_own:planner._validate_attack_plan=old_own
                else:del planner._validate_attack_plan
            planner.last_trace['temporal_candidate_protocol']=dict(version='temporal_candidate_prototype_20261029_v1',
                fractional_interaction=self.fractional_interaction,velocity_continuation=self.velocity_continuation,motion_evidence=self.motion_evidence,
                semantic_interaction_radius_m=15.,fractional_semantic_events=self.fractional_events,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                background_dynamics_and_diffusion_unchanged=True,original_off_path_preserved=True,closed_loop_gain_unverified=True)
