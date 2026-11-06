"""实验专用真实传输预算：回退和修复共用同一计数，不改公共规划器默认机制。"""
import json
from pathlib import Path


class TransportBudgetExhausted(ValueError):
    """预算未允许发送新请求；本身不代表模型返回了服务错误或无效JSON。"""


class PhysicalCallBudget:
    def __init__(self,limit=4,log_path=None):
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=30:
            raise ValueError('transport budget must be an integer in [1,30]')
        self.limit=limit;self.log_path=Path(log_path) if log_path is not None else None
        self.reset(None)

    def reset(self,scene_index):
        self.scene_index=scene_index;self.calls=0;self.blocked=0;self.outcomes=[]

    def snapshot(self):
        return dict(scenario_index=self.scene_index,transport_calls=self.calls,
                    transport_limit=self.limit,blocked_sends=self.blocked,
                    counts='all actual SDK create attempts, including quota/404/retries/repairs')

    def attach(self,planner):
        """应在API审计层安装后调用；预算拒绝不会被误记成已发送的HTTP请求。"""
        original=planner.client.chat.completions.create
        def guarded(*args,**request):
            if self.calls>=self.limit:
                self.blocked+=1
                if self.log_path is not None:
                    record=dict(kind='transport_budget_exhausted',model_not_sent=request.get('model'),**self.snapshot())
                    with self.log_path.open('a',encoding='utf8') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                raise TransportBudgetExhausted('transport_budget_exhausted_before_send')
            self.calls+=1
            outcome=dict(model=request.get('model'),returned=False,error=None)
            self.outcomes.append(outcome)
            try:
                result=original(*args,**request);outcome['returned']=True;return result
            except Exception as error:
                # 只在内存中保留原异常以维持服务失败语义，绝不序列化其正文或请求头。
                outcome['error']=error;raise
        planner.client.chat.completions.create=guarded
        planner._physical_call_budget=self


def bounded_model_request(planner,original,*args,**kwargs):
    """未发送模型不记为已尝试、不注释；真实服务失败仍照实上报。"""
    budget=planner._physical_call_budget
    start=len(budget.outcomes)
    try:return original(planner,*args,**kwargs)
    except TransportBudgetExhausted:
        actual=budget.outcomes[start:]
        planner.last_attempted_model_names=[x['model'] for x in actual]
        planner.last_attempted_model_name=actual[-1]['model'] if actual else None
        if actual and not any(x['returned'] for x in actual):
            error=actual[-1]['error']
            if error is not None and planner._is_service_failure(error):raise error
        raise
    finally:
        trace=getattr(planner,'last_trace',None)
        if isinstance(trace,dict):trace['transport_budget']=budget.snapshot()


def install_physical_budget(output,limit=4):
    """仅在独立实验进程安装；调用方先装API审计层，结束后按逆序恢复。"""
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from experiments.comparison import run_e1_group as runner
    output=Path(output);log=output.with_name(output.name+'_budget_events.jsonl')
    if log.exists():raise FileExistsError(log)
    original_init=LLMAdversarialPlanner.__init__;original_request=LLMAdversarialPlanner._request_attack_plan
    original_episode=runner.run_episode;original_generator=runner.AdversarialScenarioGenerator

    def init(planner,*args,**kwargs):
        original_init(planner,*args,**kwargs)
        if planner.provider!='dashscope':raise ValueError('budgeted model-pool experiment requires Bailian')
        if planner.client is not None:PhysicalCallBudget(limit,log).attach(planner)

    def request(planner,*args,**kwargs):
        if not hasattr(planner,'_physical_call_budget'):return original_request(planner,*args,**kwargs)
        return bounded_model_request(planner,original_request,*args,**kwargs)

    def episode(env,policy,index,global_index,*args,**kwargs):
        budget=getattr(env.llm_planner,'_physical_call_budget',None)
        if budget is not None:budget.reset(int(global_index))
        return original_episode(env,policy,index,global_index,*args,**kwargs)

    class BudgetedGenerator(original_generator):
        def step(self,env,current_t):
            budget=getattr(self.llm_planner,'_physical_call_budget',None)
            if budget is not None and budget.calls>=budget.limit:self._attacks_enabled=False
            # 原step仍结算/清除到期意图，预算不能把旧攻击无限延长。
            result=super().step(env,current_t)
            if budget is not None and budget.calls>=budget.limit:self._attacks_enabled=False
            return result

    LLMAdversarialPlanner.__init__=init;LLMAdversarialPlanner._request_attack_plan=request
    runner.run_episode=episode;runner.AdversarialScenarioGenerator=BudgetedGenerator
    def restore():
        LLMAdversarialPlanner.__init__=original_init;LLMAdversarialPlanner._request_attack_plan=original_request
        runner.run_episode=original_episode;runner.AdversarialScenarioGenerator=original_generator
    return restore
