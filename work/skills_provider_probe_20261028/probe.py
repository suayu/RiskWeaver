"""独立真实服务诊断：每模型最多一次SDK，不带场景，不读取或记录凭据正文。"""
import json
from pathlib import Path
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0] = [str(ROOT), str(ROOT/'work/skills_provider_compat_20261028')]
from provider_control import install_control
from policies.llm_adversarial_planner import LLMAdversarialPlanner


def main():
    out = Path(__file__).parent/'result.json'
    if out.exists():
        raise FileExistsError('preserve_probe_result')
    records=[]
    restore=install_control('service_only')
    try:
        for model in ['glm-5.3','qwen3.8-2.4t-a95b']:
            record=dict(model=model,provider='dashscope',physical_requests=0,SDK_max_retries=0)
            planner=LLMAdversarialPlanner(model_name=model,model_names=[model],provider='dashscope')
            if not planner.available:
                raise RuntimeError('configured_bailian_client_unavailable')
            create=planner.client.chat.completions.create
            def counted(*args,**kwargs):
                if record['physical_requests']>=1:
                    raise RuntimeError('diagnostic_budget_exceeded')
                record['physical_requests']+=1
                record['enable_thinking']=kwargs.get('extra_body',{}).get('enable_thinking')
                start=time.monotonic()
                try:
                    response=create(*args,**kwargs)
                    usage=getattr(response,'usage',None)
                    record['usage']=usage.model_dump() if usage is not None else None
                    return response
                finally:record['http_seconds']=time.monotonic()-start
            planner.client.chat.completions.create=counted
            try:
                _, data=planner._request_attack_plan('Return exactly one JSON object: {"status":"ok"}. This is a service parameter check, not a scene or attack task.')
                record.update(parsed_response=data,passed=data=={'status':'ok'})
            except Exception as error:
                # 只保存异常类型、状态和公开错误码；不保存请求头、连接配置或凭据。
                record.update(passed=False,error_type=type(error).__name__,status_code=getattr(error,'status_code',None),code=getattr(error,'code',None))
            records.append(record)
            out.write_text(json.dumps(dict(records=records,complete=len(records)==2,scene_attempts=0,goal_complete=False),ensure_ascii=False,indent=2),encoding='utf-8')
    finally:restore()
    print(json.dumps(dict(records=records,complete=True,scene_attempts=0),ensure_ascii=False))


if __name__=='__main__':
    main()
