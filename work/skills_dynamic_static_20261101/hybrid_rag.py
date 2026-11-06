"""静态攻击作为独立事件记录；保留动态attack字段与实际暴露帧的原有语义。"""
from contextlib import contextmanager
from policies.rag_runtime import AttackRAGRuntime


class HybridAttackRAGRuntime(AttackRAGRuntime):
    def mark_obstacles(self,created):
        # 真实后端返回StaticObstacle对象，查询接口才返回字典；两者都规范为真实ID。
        if self.pending is not None:
            ids=[str(item['id']) if isinstance(item,dict) else str(item.obstacle_id) for item in created]
            self.pending['obstacle_created_ids']=ids
            if ids:self.pending['accepted_intent']=True

    def planned(self,plan,trace,service_failure=False):
        super().planned(plan,trace,service_failure)
        if self.pending is not None and plan and plan.get('obstacle_plan'):
            self.pending.update(requested_attack=True,attack_kind='static',
                strategy='static_obstruction',attack_family='static_route_obstruction')
        elif self.pending is not None:
            self.pending['attack_kind']='dynamic' if plan and plan.get('attack') else 'none'


@contextmanager
def install():
    import policies.rag_runtime as module
    previous=module.AttackRAGRuntime;module.AttackRAGRuntime=HybridAttackRAGRuntime
    try:yield
    finally:module.AttackRAGRuntime=previous
