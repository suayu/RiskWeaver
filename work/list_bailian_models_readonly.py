"""仅列出与仓库活动名单相交的公开模型名；不发起推理、不打印凭据。"""
from pathlib import Path
import json
import os
import sys
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT))
from openai import OpenAI
from policies.llm_adversarial_planner import LLMAdversarialPlanner


def main():
    # 鉴权只在已授权服务器进程内使用，不写入结果文件或响应。
    client=OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'],
                  base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                  timeout=20,max_retries=0)
    disabled=LLMAdversarialPlanner._exhausted_models_from_source()
    allowed=[n for n in LLMAdversarialPlanner.FREE_MODEL_POOL if n not in disabled]
    result=dict(checked_at=time.time(),inference_calls=0,
                scope='GET /models metadata only; availability does not establish free quota or protection')
    try:
        available={model.id for model in client.models.list().data}
        result.update(status='metadata_received',active_listed_models=[n for n in allowed if n in available],
                      active_but_not_listed=[n for n in allowed if n not in available])
    except Exception as error:
        result.update(status='metadata_unavailable',error_type=type(error).__name__,
                      status_code=getattr(error,'status_code',None))
    finally:
        client.close()
    path=ROOT/'work/skills_bailian_model_metadata_20261020.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
