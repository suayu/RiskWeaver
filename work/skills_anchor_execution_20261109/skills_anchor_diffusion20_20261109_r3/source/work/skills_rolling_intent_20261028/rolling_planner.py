"""默认关闭的未来参照/滚动意图装配；不修改扩散，也不更换原语义验证器。"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'work/skills_quantity_evidence_20261028')]
from quantity_planner import QuantityEvidencePlanner
from policies.intent_validation import validate_world_intent
from rolling_facts import AttackMemory,ego_hypotheses,future_card,bounded_future_proposals,remaining_intent_candidate
from temporal_audit import continuity


class RollingIntentPlanner(QuantityEvidencePlanner):
    def __init__(self,sim_cfg,call_budget=4,extra_recipes=False,rolling_intent=False):
        if not isinstance(rolling_intent,bool):raise ValueError('rolling_intent_requires_boolean')
        super().__init__(sim_cfg,call_budget,extra_recipes,quantity_evidence=False)
        if rolling_intent and (not extra_recipes or self.interval!=1. or self.dt!=.1):
            raise ValueError('prototype_requires_existing_recipes_and_10hz_1s_anchor_contract')
        self.rolling_intent=rolling_intent;self.memory=AttackMemory();self.rolling_audit=[]

    def reset_episode(self):
        super().reset_episode();self.memory.clear()

    def expanded_candidates(self,env,state,planner):
        existing=super().expanded_candidates(env,state,planner)
        if not self.rolling_intent:return existing
        lanes=[np.asarray(line)[:,:2] for line in env.scenario_dict.get('lanes',[]) if len(line)>1]
        states=np.asarray(env.data_dict['agent'][-1],float);proposals=[]
        previous=self.previous_status.get('previous')
        for actor in state['agents']:
            target=actor['id']
            if not 0<=target<len(states) or not env.agent_active[target]:continue
            proposals+=bounded_future_proposals(env.ego_state,states[target],target,self.forecast,previous=previous,dt=self.dt)
        # 仅从同一目标的已存在意图构造延续版本；换车不强行接到旧攻击上。
        if previous:
            for c in existing[:]:
                v=remaining_intent_candidate(previous,c,states[c['target_id']],int(env.current_step),self.dt)
                if v:proposals.append(v)
        output=[]
        for index,proposal in enumerate(proposals):
            proposal=dict(proposal,candidate_id=f'rolling_{index}')
            estimates,reason=self.builder.screen(env,proposal['target_id'],np.asarray(proposal['anchors']),lanes)
            record=dict(candidate=proposal,screen_rejection=reason,accepted=False)
            if reason is None:
                semantic=validate_world_intent(planner,state,proposal);record['semantic_validation']=semantic
                if semantic['accepted']:
                    output.append(dict(proposal,**estimates,semantic_evidence=semantic['metrics'],screened=True));record['accepted']=True
            self.rolling_audit.append(record)
        return existing+output

    def build_context(self,env,state,planner,profile_context=None):
        if not self.rolling_intent:return super().build_context(env,state,planner,profile_context)
        self.forecast=ego_hypotheses(state,int(env.current_step),self.dt)
        self.previous_status=self.memory.inspect(env.current_scene_id,int(env.current_step),env.agent_active,self.dt)
        self.rolling_audit=[]
        context=super().build_context(env,state,planner,profile_context)
        previous=self.previous_status.get('previous');cards={}
        for c in context['candidates']:
            card=future_card(c,self.forecast,self.dt)
            if previous:
                actor=np.asarray(env.data_dict['agent'][-1])[c['target_id']]
                card['continuity']=continuity(previous,dict(source_step=int(env.current_step),target_id=c['target_id'],anchors=c['anchors']),actor[:2],self.dt)
            cards[c['candidate_id']]=card
        context.update(ego_future_reference=self.forecast,rolling_intent=dict(previous=self.previous_status,
            candidate_facts=cards,proposal_audit=self.rolling_audit,response_schema_unchanged=True,
            sparse_intents_only=True,original_semantic_CV_precheck_retained=True,
            probabilities_and_actual_policy_future_not_available=True))
        return context

    def plan(self,planner,env,state,*args,**kwargs):
        if not self.rolling_intent:return super().plan(planner,env,state,*args,**kwargs)
        request=planner._request_attack_plan
        def decorated(prompt,scene_image=None):
            prefix,raw=prompt.rsplit('\n',1);payload=json.loads(raw)
            if 'rolling_intent' not in payload['context']:raise ValueError('rolling_context_missing_from_real_prompt')
            # 完整提案拒绝日志保留在trace；模型只需看发布目录及其事实，避免重复膨胀。
            payload['context']['rolling_intent'].pop('proposal_audit',None)
            skill=(HERE/'SKILL.md').read_text(encoding='utf-8').split('---',2)[-1]
            return request(prefix+'\n'+skill+'\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False),scene_image=scene_image)
        planner._request_attack_plan=decorated;result=None
        try:
            result=super().plan(planner,env,state,*args,**kwargs)
            if hasattr(self,'forecast'):
                receipt=self.memory.commit(env.current_scene_id,int(env.current_step),result,self.forecast,self.previous_status,self.dt)
                planner.last_trace['rolling_episode_update']=receipt
            return result
        finally:
            planner._request_attack_plan=request
            planner.last_trace['rolling_intent_protocol']=dict(enabled=True,version='rolling_future_prototype_20261029_v1',
                skill_sha256=hashlib.sha256((HERE/'SKILL.md').read_bytes()).hexdigest(),
                candidate_catalog_changed=True,response_schema_unchanged=True,safety_and_diffusion_changed=False,
                memory_is_accepted_not_executed=True,new_API_budget=0,closed_loop_performance_verified=False)
