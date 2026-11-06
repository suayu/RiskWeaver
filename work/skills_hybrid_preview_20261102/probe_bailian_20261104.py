import json,sys
from urllib.parse import urlsplit
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from policies.llm_adversarial_planner import LLMAdversarialPlanner
models=['qwen-mt-uni','qwen3.8-omni-flash','qwen3.8-omni-flash-realtime']
client=LLMAdversarialPlanner(provider='dashscope',model_name='deepseek-v4.1-flash-1017',model_names=['deepseek-v4.1-flash-1017']).client
assert client and urlsplit(str(client.base_url)).hostname=='dashscope.aliyuncs.com'
client=client.with_options(timeout=25,max_retries=0)
for name in models:
 row=dict(model=name)
 try:
  r=client.chat.completions.create(model=name,messages=[dict(role='user',content='Return exactly this JSON object: {"probe":true}')],response_format={'type':'json_object'},extra_body={'enable_thinking':False},max_tokens=32)
  row.update(returned_model=r.model,content=r.choices[0].message.content,finish=r.choices[0].finish_reason)
 except Exception as e:
  body=getattr(e,'body',None)
  error=body.get('error',body) if isinstance(body,dict) else {}
  row.update(http=getattr(e,'status_code',None),error_code=error.get('code'),error_type=type(e).__name__)
 print(json.dumps(row,ensure_ascii=False),flush=True)
