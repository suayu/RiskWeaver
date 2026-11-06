"""独立默认关闭工厂：让目录构建和最终复查使用同一个路线后备规则。"""
import copy
import hashlib
from pathlib import Path

from adapter import install


def make_static(sim_cfg, call_budget=4, enabled=False):
    from policies.hybrid_skills_planner import StaticCombinationPlanner
    if not isinstance(enabled,bool):raise ValueError('route_fallback_flag_requires_bool')
    if not enabled:return StaticCombinationPlanner(sim_cfg,call_budget)

    class RouteStaticPlanner(StaticCombinationPlanner):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            original=self.registry.load
            def load(*args,**kwargs):
                knowledge=copy.deepcopy(original(*args,**kwargs))
                if self.registry.level in ('analysis','full'):
                    raw=Path(__file__).with_name('SKILL.md').read_bytes()
                    knowledge['documents'].append(dict(path='prototype/rw-route-static-clearance/SKILL.md',
                        sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf-8')))
                    knowledge['version']+='+route-static-fallback-prototype-v1'
                return knowledge
            self.registry.load=load

        def build_context(self,*args,**kwargs):
            context=super().build_context(*args,**kwargs)
            context['static_clearance_policy']=dict(version='original_forward_or_route_footprint_entry_v1',
                original_forward_minimum_m=15,route_footprint_entry_minimum_m=15,
                initial_ego_OBB_margin_m=self.builder.margin,
                original_success_path_preserved=True,background_safety_unchanged=True,
                not_executed_or_dynamic_reachability_or_evasion_certificate=True)
            return context

        def plan(self,planner,env,*args,**kwargs):
            with install(planner,env,enabled=True,initial_margin=self.builder.margin) as receipt:
                try:
                    return super().plan(planner,env,*args,**kwargs)
                finally:
                    if isinstance(getattr(planner,'last_trace',None),dict):
                        # 全部接受/拒绝几何证据保留trace，不把长串诊断塞入模型提示。
                        planner.last_trace['route_static_clearance']=copy.deepcopy(receipt)

    return RouteStaticPlanner(sim_cfg,call_budget)
