"""独立研究原型：以真实规划API预算约束模型，空目录仍可低频继续筛查。"""
from scenario_generator import AdversarialScenarioGenerator


class EventBudgetGenerator(AdversarialScenarioGenerator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        settings=getattr(getattr(self.cfg.sim,'llm',None),'skills',None)
        self.event_budget_enabled=bool(getattr(settings,'event_budget_enabled',False))
        self.max_screening_queries=int(getattr(settings,'max_screening_queries',40))
        self.screening_query_count=0
        if self.event_budget_enabled and self.skills_planner is None:
            raise ValueError('event budget requires explicit skills pipeline')
        if not 1<=self.max_screening_queries<=80:raise ValueError('invalid screening budget')

    def reset_episode_stats(self,*args,**kwargs):
        super().reset_episode_stats(*args,**kwargs)
        self.screening_query_count=0

    def step(self,env,current_t):
        if not getattr(self,'event_budget_enabled',False):return super().step(env,current_t)
        before_queries=self._llm_call_times
        before_enabled=self._attacks_enabled
        result=super().step(env,current_t)
        if self._llm_call_times==before_queries:return result
        self.screening_query_count+=1
        # 当前固定单模型实验里engine.calls等于API请求数；多模型恢复需另计传输尝试。
        self._llm_call_times=self.skills_planner.calls
        if self._llm_call_times>=self._max_call_times or self.screening_query_count>=self.max_screening_queries:
            self._attacks_enabled=False
        elif before_enabled and not bool(getattr(self.llm_planner,'last_request_failed',False)):
            # 仅撤销“空目录被当成一次API消耗”的预算熔断，不撤销服务失败熔断。
            self._attacks_enabled=True
        trace=getattr(self,'query_trace',None)
        if trace is not None:
            trace.append(dict(kind='skills_budget_accounting',step=int(current_t),
                screening_queries=self.screening_query_count,max_screening_queries=self.max_screening_queries,
                actual_model_calls=self.skills_planner.calls,max_model_calls=self._max_call_times,
                budget_basis='single-model requests including schema repairs; not candidate screening'))
        return result
