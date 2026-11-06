"""百炼Omni共同参数装配；保留原提供方能力参数，不沿用Qwen专用覆盖。"""
from contextlib import contextmanager
import hashlib
import json
import time

MODEL = 'qwen3.8-omni-flash'


@contextmanager
def install(directory, planner_class=None):
    if planner_class is None:
        from policies.llm_adversarial_planner import LLMAdversarialPlanner
        planner_class = LLMAdversarialPlanner
    original = planner_class.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.provider != 'dashscope' or self.client is None:
            return
        self.client = self.client.with_options(timeout=180, max_retries=0)
        create = self.client.chat.completions.create
        def bounded(*args, **kwargs):
            if kwargs.get('model') != MODEL:
                raise ValueError('unfrozen_model_transport_parameters')
            # 保留通过文本探测的原请求参数；原请求参数原样保留。
            values = dict(kwargs, max_tokens=8192)
            record = dict(model=MODEL, max_tokens=8192, timeout_seconds=180, SDK_retries=0,
                          reasoning_effort=values.get('reasoning_effort'),
                          extra_body=values.get('extra_body'), started_at=time.time(),
                          messages_sha256=hashlib.sha256(json.dumps(values['messages'], ensure_ascii=False,
                                                                  sort_keys=True).encode()).hexdigest())
            start = time.monotonic()
            try:
                return create(*args, **values)
            except Exception as error:
                record.update(error_type=type(error).__name__, status_code=getattr(error, 'status_code', None))
                raise
            finally:
                record['elapsed_seconds'] = time.monotonic() - start
                directory.mkdir(parents=True, exist_ok=True)
                with (directory/'common_transport.jsonl').open('a', encoding='utf-8') as file:
                    file.write(json.dumps(record, ensure_ascii=False)+'\n')
        self.client.chat.completions.create = bounded
    planner_class.__init__ = initialize
    try:
        yield
    finally:
        planner_class.__init__ = original
