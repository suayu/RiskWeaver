"""仅实验进程内固定共同推理参数；不改变原默认入口、回复或请求次数。"""
from contextlib import contextmanager
import hashlib,json,time


@contextmanager
def install(directory,planner_class=None):
    if planner_class is None:
        from policies.llm_adversarial_planner import LLMAdversarialPlanner
        planner_class=LLMAdversarialPlanner
    original=planner_class.__init__
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs)
        if self.provider!='dashscope' or self.client is None:return
        self.client=self.client.with_options(timeout=180,max_retries=0)
        create=self.client.chat.completions.create
        def bounded(*args,**kwargs):
            model=kwargs.get('model')
            if model not in ('qwen3.8-2.4t-a95b','glm-5.3'):
                raise ValueError('unfrozen_model_transport_parameters')
            values=dict(kwargs,max_tokens=8192,reasoning_effort='low')
            values['extra_body']=dict(values.get('extra_body') or {},enable_thinking=True)
            record=dict(model=model,max_tokens=8192,reasoning_effort='low',timeout_seconds=180,
                SDK_retries=0,extra_body=values['extra_body'],started_at=time.time(),
                messages_sha256=hashlib.sha256(json.dumps(values['messages'],ensure_ascii=False,sort_keys=True).encode()).hexdigest())
            start=time.monotonic()
            try:return create(*args,**values)
            except Exception as error:
                record.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None));raise
            finally:
                record['elapsed_seconds']=time.monotonic()-start
                directory.mkdir(parents=True,exist_ok=True)
                with (directory/'common_transport.jsonl').open('a',encoding='utf-8') as file:
                    file.write(json.dumps(record,ensure_ascii=False)+'\n')
        self.client.chat.completions.create=bounded
    planner_class.__init__=initialize
    try:yield
    finally:planner_class.__init__=original
