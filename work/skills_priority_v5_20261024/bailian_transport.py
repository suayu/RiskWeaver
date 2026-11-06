"""新实验的公共提供方参数适配；旧/新组可共同启用，不改变策略选择或请求数。"""
import copy


def compatible_request(provider, kwargs):
    result = copy.deepcopy(kwargs)
    # 仅依据本次真实400回执处理精确模型名，不推断其他GLM或平台的能力。
    if provider == 'dashscope' and result.get('model') == 'glm-5.3':
        result.setdefault('extra_body', {})['enable_thinking'] = True
    return result


def install_provider_compatibility():
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    original = LLMAdversarialPlanner._request_attack_plan_once
    def once(planner, model_name, prompt, scene_image=None):
        if planner.provider != 'dashscope' or model_name != 'glm-5.3':
            return original(planner, model_name, prompt, scene_image=scene_image)
        completions = planner.client.chat.completions
        create = completions.create
        def adjusted(*args, **kwargs):
            values = compatible_request(planner.provider, kwargs)
            trace = getattr(planner, 'last_trace', None)
            if isinstance(trace, dict):
                trace.setdefault('provider_parameter_adaptations', []).append(dict(
                    model=model_name, enable_thinking=True, additional_requests=0))
            return create(*args, **values)
        completions.create = adjusted
        try:
            return original(planner, model_name, prompt, scene_image=scene_image)
        finally:
            completions.create = create
    LLMAdversarialPlanner._request_attack_plan_once = once
    def restore():
        LLMAdversarialPlanner._request_attack_plan_once = original
    return restore
