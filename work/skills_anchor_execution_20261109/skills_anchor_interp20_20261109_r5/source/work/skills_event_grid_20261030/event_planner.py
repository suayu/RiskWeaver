"""独立默认关闭的未来事件配方；继承原时序技能/调用预算/语义验证。"""
from pathlib import Path
import hashlib,json,sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'work/skills_temporal_candidates_20261029'))
from temporal_planner import TemporalCandidatePlanner
from policies.intent_validation import validate_world_intent
from event_grid import screened_event_candidates,TIMES

def knowledge():
    # SKILL.md只作为提示知识读取，不执行其中代码；文件纳入新协议源码哈希。
    return (HERE/'SKILL.md').read_text(encoding='utf-8').split('---',2)[-1].strip()


class FutureEventPlanner(TemporalCandidatePlanner):
    def __init__(self,*args,event_grid=False,**kwargs):
        if not isinstance(event_grid,bool):raise ValueError('event_grid_requires_boolean')
        super().__init__(*args,**kwargs)
        if event_grid and not (self.extra_recipes and self.rolling_intent and self.fractional_interaction):raise ValueError('event_grid_requires_temporal_expansion_path')
        self.event_grid_enabled=event_grid;self.event_grid_audit=[]

    def expanded_candidates(self,env,state,planner):
        original=super().expanded_candidates(env,state,planner)
        if not self.event_grid_enabled:return original
        additions,self.event_grid_audit=screened_event_candidates(env,state,self.builder,lambda snapshot,candidate:validate_world_intent(planner,snapshot,candidate))
        return original+additions

    def build_context(self,env,state,planner,profile_context=None):
        self.event_grid_audit=[]
        context=super().build_context(env,state,planner,profile_context)
        if not self.event_grid_enabled:return context
        visible={c['candidate_id']:dict(target_id=c['target_id'],event_time_seconds=c['event_time_seconds'],
            event_within_current_lease=c['event_within_current_lease'],interpolation_event_residual_m=c['interpolation_event_residual_m'],
            required_sparse_acceleration_norm_mps2=c['required_sparse_acceleration_norm_mps2'],
            event_reference=c['event_reference'],not_actual_execution_or_inevitable_collision=True)
            for c in context['candidates'] if c.get('recipe')=='full_grid_source_CV_sparse_intercept_v1'}
        context['future_event_evidence']=dict(grid_seconds=list(TIMES),max_published_per_actor=2,visible_candidates=visible,
            unchanged_sparse_coefficient_cap_mps2=3.8,actual_background_safety_and_dynamics_require_final_validation=True)
        return context

    def plan(self,planner,*args,**kwargs):
        if not self.event_grid_enabled:return super().plan(planner,*args,**kwargs)
        request=planner._request_attack_plan
        def decorate(prompt,scene_image=None):
            prefix,raw=prompt.rsplit('\n',1);payload=json.loads(raw)
            if 'future_event_evidence' not in payload['context']:raise ValueError('future_event_facts_missing_in_actual_request')
            return request(prefix+'\n'+knowledge()+'\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False),scene_image=scene_image)
        planner._request_attack_plan=decorate
        try:return super().plan(planner,*args,**kwargs)
        finally:
            planner._request_attack_plan=request
            planner.last_trace['future_event_protocol']=dict(enabled=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                skill_sha256=hashlib.sha256((HERE/'SKILL.md').read_bytes()).hexdigest(),
                grid_audit=self.event_grid_audit,grid_seconds=list(TIMES),source_future_only=True,
                no_extra_API_or_diffusion_samples=True,background_dynamics_thresholds_unchanged=True,
                actual_effect_or_majority_correct_planning_certified=False)
