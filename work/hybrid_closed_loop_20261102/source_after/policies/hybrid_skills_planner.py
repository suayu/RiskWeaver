"""车辆攻击优先，确无可用动态目录时才调用静态组合技能；旧路径由开关保留。"""
import copy
import hashlib
import json
from pathlib import Path

from policies.ego_profile import scene_conditions
from policies.skills_planner import SkillsAttackPlanner
from policies.static_attack_catalog import build_static_combinations,validate_placements


class StaticCombinationPlanner(SkillsAttackPlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        load=self.registry.load
        def with_routing_skill(*args,**kwargs):
            knowledge=load(*args,**kwargs)
            if self.registry.level in ('analysis','full'):
                root=Path(__file__).resolve().parents[1]/'skills/riskweaver'
                raw=(root/'rw-static-fallback/SKILL.md').read_bytes()
                manifest=json.loads((root/'manifest.json').read_text())
                digest=hashlib.sha256(raw).hexdigest()
                if digest!=manifest['files'].get('rw-static-fallback/SKILL.md'):
                    raise ValueError('static_fallback_skill_hash_changed')
                knowledge['documents'].append(dict(path='rw-static-fallback/SKILL.md',sha256=digest,text=raw.decode('utf-8')))
                knowledge['version']+='+static-fallback-v1'
            return knowledge
        self.registry.load=with_routing_skill

    def build_context(self,env,state,planner,profile_context=None):
        candidates,audit=build_static_combinations(env,planner,self.builder)
        self.environment=env;self.current_planner=planner
        self.published_obstacle_ids={row['obstacle_candidate_id'] for row in candidates}
        return dict(profile=None,history=None,candidates=[],obstacle_candidates=candidates,
            static_candidate_audit=audit,scene_conditions=scene_conditions(env),profile_enabled=False,
            candidate_expansion_enabled=False,source_step=int(env.current_step),
            scene_id=str(env.current_scene_id),allow_dynamic=False,
            routing_reason='no_usable_dynamic_vehicle_candidate; choose_static_combination',
            theoretical_positive_area_not_escape_guarantee=True)

    def validate_response(self,response,*args):
        super().validate_response(response,*args)
        if response.get('obstacle_candidate_id') is None:
            raise ValueError('static_fallback_requires_catalog_selection')
        if response['obstacle_candidate_id'] not in self.published_obstacle_ids:
            raise ValueError('static_choice_not_in_catalog')

    def select_obstacle(self,response,context,ranking):
        selected=response.get('obstacle_candidate_id')
        item=next((row for row in context.get('obstacle_candidates',[])
                   if row['obstacle_candidate_id']==selected),None)
        if item is None:raise ValueError('static_choice_not_in_catalog')
        result=validate_placements(self.environment,self.current_planner,self.builder,item['placements'])
        if not result['accepted']:raise ValueError('static_final_revalidation:'+result['reason'])
        return copy.deepcopy(item['placements'])


class DynamicFirstStaticPlanner:
    def __init__(self,dynamic,sim_cfg,call_budget=4):
        self.dynamic=dynamic;self.static=StaticCombinationPlanner(sim_cfg,call_budget)
        self.budget=int(call_budget)
        # 外层与内层是不同规划器，RAG必须绑定最终外层，不能通过属性代理误判为已绑定。
        self._rag_bound_runtime=None;self.rag_runtime=None

    @property
    def calls(self):return self.dynamic.calls+self.static.calls

    def __getattr__(self,name):return getattr(self.dynamic,name)

    def reset_episode(self):
        self.dynamic.reset_episode();self.static.reset_episode()

    def plan(self,planner,env,state,instruction,profile_context=None,scene_image=None):
        original_use=planner.use_obstacles;original_allow=getattr(planner,'allow_trajectory',True)
        planner.use_obstacles=False;planner.allow_trajectory=original_allow
        try:
            context=(self.dynamic.build_context(env,state,planner,profile_context) if original_allow
                else dict(candidates=[],semantic_precheck=[],routing_reason='dynamic_disabled_by_attack_mode'))
        finally:
            planner.use_obstacles=original_use;planner.allow_trajectory=original_allow
        use_dynamic=bool(context.get('candidates'))
        selected=self.dynamic if use_dynamic else self.static
        if self.calls>=self.budget:
            planner.last_trace=dict(plan_validation='skills_api_budget_exhausted',mode='dynamic_first_static_fallback')
            return dict(attack=False,attack_target_id=-1,strategy='none',anchors=[],reason='skills_api_budget_exhausted')
        if getattr(self,'rag_runtime',None) is not None:selected.rag_runtime=self.rag_runtime
        build=selected.build_context;old_budget=selected.budget
        if use_dynamic:selected.build_context=lambda *a,**k:copy.deepcopy(context)
        selected.budget=selected.calls+(self.budget-self.calls)
        planner.use_obstacles=not use_dynamic;planner.allow_trajectory=use_dynamic
        try:
            plan=selected.plan(planner,env,state,instruction,profile_context if use_dynamic else None,scene_image)
            if plan and plan.get('obstacle_plan'):
                plan.update(attack_kind='static',static_attack=True,attack_family='static_route_obstruction')
            elif plan and plan.get('attack'):plan['attack_kind']='dynamic'
            return plan
        finally:
            selected.build_context=build;selected.budget=old_budget
            planner.use_obstacles=original_use;planner.allow_trajectory=original_allow
            if isinstance(getattr(planner,'last_trace',None),dict):
                planner.last_trace['hybrid_routing']=dict(version='dynamic_first_static_v1_20261101',
                    branch='dynamic' if use_dynamic else 'static',dynamic_candidate_count=len(context.get('candidates',[])),
                    dynamic_precheck=copy.deepcopy(context.get('semantic_precheck',[])),
                    actual_total_logical_calls=self.calls,call_budget=self.budget,
                    service_failures_are_not_converted_to_no_vehicle=True,
                    accepted_plan_is_not_actual_execution_or_threat=True)


def create_planner(dynamic_factory,sim_cfg,call_budget=4,**kwargs):
    """新参数缺省关闭；开启时仅给内层动态技能关闭障碍字段，整体静态能力仍保留。"""
    if not bool(getattr(sim_cfg.llm.skills,'static_fallback_enabled',False)):
        return dynamic_factory(sim_cfg,call_budget,**kwargs)
    from omegaconf import OmegaConf
    dynamic_cfg=OmegaConf.create(OmegaConf.to_container(sim_cfg,resolve=True))
    dynamic_cfg.llm.obstacles.enabled=False
    dynamic=dynamic_factory(dynamic_cfg,call_budget,**kwargs)
    return DynamicFirstStaticPlanner(dynamic,sim_cfg,call_budget)
