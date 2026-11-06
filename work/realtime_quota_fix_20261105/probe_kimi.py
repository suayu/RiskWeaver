"""固定百炼新模型单次小请求，只核验参数兼容，不进行场景仿真。"""
import json,time,os,sys
from pathlib import Path
R=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(R))
from policies.llm_adversarial_planner import LLMAdversarialPlanner
p=LLMAdversarialPlanner(model_name='kimi-k3.1-pro',model_names=['kimi-k3.1-pro'],provider='dashscope')
start=time.monotonic();out=dict(model='kimi-k3.1-pro',physical_requests=1,max_tokens=256,paid_fallback=False)
try:
 r=p.client.with_options(timeout=90,max_retries=0).chat.completions.create(model='kimi-k3.1-pro',messages=[{'role':'user','content':'Return JSON only: {"ok": true}'}],max_tokens=256,response_format={'type':'json_object'},extra_body={'enable_thinking':False})
 out.update(status='success',returned_model=r.model,usage=r.usage.model_dump(),finish_reason=r.choices[0].finish_reason,valid_json=json.loads(r.choices[0].message.content)=={'ok':True})
except Exception as e:
 body=getattr(e,'body',{})
 if not isinstance(body,dict):body={}
 error=body.get('error',body);out.update(status='error',error_type=type(e).__name__,status_code=getattr(e,'status_code',None),code=error.get('code') if isinstance(error,dict) else None,message=str(error.get('message',''))[:400] if isinstance(error,dict) else '')
out['elapsed_seconds']=time.monotonic()-start
(R/'work/realtime_quota_fix_20261105/kimi_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
