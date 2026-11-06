"""新实验共用的百炼传输适配；不覆盖历史入口或修改技能、扩散和安全。"""
import copy

THINKING_REQUIRED = frozenset({'glm-5.3', 'qwen3.8-2.4t-a95b'})


def request_parameters(provider, kwargs):
    """只依据已留存的精确模型400回执改参，不推断其他模型能力。"""
    values = copy.deepcopy(kwargs)
    if provider == 'dashscope' and values.get('model') in THINKING_REQUIRED:
        values.setdefault('extra_body', {})['enable_thinking'] = True
    return values


def install_control(model_policy='legacy'):
    """仅当前独立进程生效；语义拒绝不换模是显式实验选项，默认保留旧行为。"""
    if model_policy not in {'legacy', 'service_only'}:
        raise ValueError('unknown_semantic_model_policy')
    from policies.llm_adversarial_planner import LLMAdversarialPlanner as Planner
    once_original = Planner._request_attack_plan_once
    advance_original = Planner._advance_after_invalid_plan

    def once(planner, model_name, prompt, scene_image=None):
        if planner.provider != 'dashscope' or model_name not in THINKING_REQUIRED:
            return once_original(planner, model_name, prompt, scene_image=scene_image)
        completions = planner.client.chat.completions
        create = completions.create

        def adjusted(*args, **kwargs):
            values = request_parameters(planner.provider, kwargs)
            trace = getattr(planner, 'last_trace', None)
            if isinstance(trace, dict):
                trace.setdefault('common_provider_adaptations', []).append(dict(
                    model=model_name, enable_thinking=True, extra_requests=0,
                    basis='preserved_exact_model_HTTP400_requires_True'))
            # 预算层仍位于真实SDK外侧；这里只传参，不补发、重试或改写错误。
            return create(*args, **values)

        completions.create = adjusted
        try:
            return once_original(planner, model_name, prompt, scene_image=scene_image)
        finally:
            completions.create = create

    def advance(planner):
        if planner.provider == 'dashscope' and model_policy == 'service_only':
            trace = getattr(planner, 'last_trace', None)
            if isinstance(trace, dict):
                trace['semantic_model_transition'] = dict(action='retain_current_model',
                    policy=model_policy, service_fallback_unchanged=True)
            return
        return advance_original(planner)

    Planner._request_attack_plan_once = once
    Planner._advance_after_invalid_plan = advance

    def restore():
        Planner._request_attack_plan_once = once_original
        Planner._advance_after_invalid_plan = advance_original
    return restore
