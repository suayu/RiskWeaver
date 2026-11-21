"""独立进程装配Realtime；关闭时保留原客户端，异常后恢复构造器。"""
from contextlib import contextmanager
from pathlib import Path
import os
from work.realtime_planner_20261104.transport import RealtimeClient, MODEL


@contextmanager
def install(output, enabled=True, planner_class=None):
    if not enabled:
        yield
        return
    if planner_class is None:
        from policies.llm_adversarial_planner import LLMAdversarialPlanner
        planner_class=LLMAdversarialPlanner
    original=planner_class.__init__
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        if self.provider!='dashscope' or tuple(self.model_names)!=(MODEL,):
            raise ValueError('fixed_approved_realtime_model_required')
        self.client=RealtimeClient(os.environ['DASHSCOPE_API_KEY'],Path(output)/'realtime_transport',timeout=180)
    planner_class.__init__=initialize
    try:
        yield
    finally:
        planner_class.__init__=original
