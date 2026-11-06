"""只读百炼模型目录；不发送生成请求，不打印或落盘凭据。"""
import datetime
import json
from pathlib import Path
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0] = [str(ROOT), str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from model_pool_protocol import inspect_model_source
from policies.llm_adversarial_planner import LLMAdversarialPlanner

APPROVED = ['qwen3.8-27b', 'kimi-k3', 'qwen3.8-max-0902', 'deepseek-v4.1-flash-1017',
            'deepseek-v4.1-flash', 'glm-5.3', 'deepseek-v4-pro-0813', 'qwen3.8-2.4t-a95b',
            'qwen3.8-max', 'qwen3.7-flash']


def main():
    source = inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    active = source['active_models']
    if not active:
        raise RuntimeError('no_active_name_for_metadata_client_initialization')
    planner = LLMAdversarialPlanner(provider='dashscope', model_name=active[0], model_names=[active[0]])
    client = planner.client.with_options(timeout=30, max_retries=0)
    pages, models = 0, []
    try:
        page = client.models.list()
        while True:
            pages += 1
            for item in page.data:
                models.append({k: getattr(item, k, None) for k in ['id', 'created', 'owned_by']})
            if not page.has_next_page():
                break
            if pages >= 10:
                raise RuntimeError('catalog_pagination_limit')
            page = page.get_next_page()
    except Exception as error:
        # 不打印SDK异常全文，避免服务端错误回显敏感请求信息。
        print(json.dumps(dict(status='metadata_failed', error_type=type(error).__name__,
            status_code=getattr(error, 'status_code', None), planning_requests=0), ensure_ascii=False))
        raise SystemExit(1)
    listed = {m['id'] for m in models}
    result = dict(status='completed', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        metadata_pages=pages, planning_requests=0, source_active_models=active,
        source_disabled_models=source['disabled_models'], catalog=models,
        approved_models=[dict(model=name, listed=name in listed, source_disabled=name in source['disabled_models'])
                         for name in APPROVED],
        listed_not_disabled=sorted(listed-set(source['disabled_models'])),
        catalog_visibility_does_not_prove_inference_permission_or_free_quota=True,
        free_quota_not_reported_by_this_metadata_response=True, model_pool_modified=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
