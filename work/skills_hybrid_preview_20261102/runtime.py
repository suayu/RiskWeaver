"""显式启用混合路由预览一致性；只调整校验作用域，不改门槛和候选算法。"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

VERSION='hybrid-preview-semantic-scope-v1-20261102'


@contextmanager
def consistent_preview(dynamic, planner):
    """目录预览临时使用plan的既有帧间校验；返回前恢复，避免正式plan嵌套装配。"""
    if getattr(dynamic,'_consistent_preview_active',False):
        raise RuntimeError('consistent_preview_already_active')
    from work.skills_temporal_candidates_20261029.temporal_intents import guarded_fractional_validator
    old_build=dynamic.build_context
    had_build='build_context' in dynamic.__dict__
    own_build=dynamic.__dict__.get('build_context')
    had_marker='_consistent_preview_active' in dynamic.__dict__
    marker=dynamic.__dict__.get('_consistent_preview_active')
    dynamic._consistent_preview_active=True
    receipt=dict(version=VERSION,previews=[],additional_API_requests=0,
        interaction_radius_m=15.,candidate_generation_and_safety_unchanged=True,
        preview_is_not_accepted_plan_or_actual_threat=True)
    def prepared(env,state,low,profile_context=None):
        if low is not planner:raise ValueError('routing_preview_planner_identity_changed')
        original=low._validate_attack_plan
        had_validator='_validate_attack_plan' in low.__dict__
        own_validator=low.__dict__.get('_validate_attack_plan')
        events=[]
        record=dict(scene_id=str(env.current_scene_id),source_step=int(env.current_step),
            fractional_semantic_events=events,status='preparing')
        receipt['previews'].append(record)
        try:
            low._validate_attack_plan=guarded_fractional_validator(original,events)
            context=old_build(env,state,low,profile_context)
            record.update(status='completed',published_candidates=len(context.get('candidates',[])),
                published_candidate_ids=[c['candidate_id'] for c in context.get('candidates',[])])
            return context
        except BaseException as error:
            record.update(status='failed',error_type=type(error).__name__)
            raise
        finally:
            if had_validator:low._validate_attack_plan=own_validator
            elif '_validate_attack_plan' in low.__dict__:del low._validate_attack_plan
    dynamic.build_context=prepared
    try:yield receipt
    finally:
        if had_build:dynamic.build_context=own_build
        else:del dynamic.build_context
        if had_marker:dynamic._consistent_preview_active=marker
        else:del dynamic._consistent_preview_active


@contextmanager
def install(enabled=False):
    """过程内开关支持已创建与新创建混合引擎；关闭时类和调用行为原样保留。"""
    if not isinstance(enabled,bool):raise ValueError('consistent_routing_preview_requires_boolean')
    receipt=dict(enabled=enabled,version=VERSION,calls=[],
        algorithm_changed=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not enabled:
        yield receipt
        return
    from policies.hybrid_skills_planner import DynamicFirstStaticPlanner
    original=DynamicFirstStaticPlanner.plan
    if getattr(original,'_consistent_preview_patch',False):
        raise RuntimeError('consistent_preview_install_already_active')
    def routed(self,planner,env,*args,**kwargs):
        if not bool(getattr(self.dynamic,'fractional_interaction',False)):
            receipt['calls'].append(dict(applied=False,reason='fractional_interaction_disabled'))
            return original(self,planner,env,*args,**kwargs)
        with consistent_preview(self.dynamic,planner) as current:
            receipt['calls'].append(current)
            try:return original(self,planner,env,*args,**kwargs)
            finally:
                if isinstance(getattr(planner,'last_trace',None),dict):
                    planner.last_trace['hybrid_preview_protocol']=current
    routed._consistent_preview_patch=True
    DynamicFirstStaticPlanner.plan=routed
    try:yield receipt
    finally:DynamicFirstStaticPlanner.plan=original
