"""短请求区分提供方响应问题与场景长输入问题；不作为场景规划或攻击效果。"""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT))
from policies.llm_adversarial_planner import LLMAdversarialPlanner


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise ValueError('CPU_only_probe')
    target=ROOT/'work/skills_service_health_20261101/result.json'
    if target.exists():raise FileExistsError('preserve_service_probe')
    rows=[]
    for model in ('glm-5.3','qwen3.8-2.4t-a95b'):
        planner=LLMAdversarialPlanner(model_name=model,model_names=[model],provider='dashscope')
        start=time.monotonic();row=dict(model=model,physical_calls=1,max_tokens=512,timeout_seconds=25)
        try:
            response=planner.client.with_options(timeout=25,max_retries=0).chat.completions.create(
                model=model,messages=[dict(role='user',content='Return JSON only: {"ok":true}.')],
                response_format=dict(type='json_object'),max_tokens=512,extra_body=dict(enable_thinking=True))
            row.update(HTTP_success=True,model_returned=response.model,content=response.choices[0].message.content,
                finish_reason=response.choices[0].finish_reason,
                usage=response.usage.model_dump() if response.usage else None)
        except Exception as error:
            row.update(HTTP_success=False,error_type=type(error).__name__,status_code=getattr(error,'status_code',None))
        row['elapsed_seconds']=time.monotonic()-start;rows.append(row)
        target.with_name('result.partial.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    target.write_text(json.dumps(dict(status='completed',requests=2,results=rows,
        scope='service_only_not_scene_performance',new_execution_frames=0,D=None),indent=2),encoding='utf-8')
    print(json.dumps(rows))
