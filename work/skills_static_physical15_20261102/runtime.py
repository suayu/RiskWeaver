"""过程内可选工厂装配；只替换新建混合引擎静态分支，退出恢复旧构造函数。"""
from contextlib import contextmanager
import inspect

from .planner import make_static, VERSION


@contextmanager
def install(enabled=False):
    if not isinstance(enabled, bool):
        raise ValueError('static_physical15_flag_requires_bool')
    receipt = dict(enabled=enabled, version=VERSION, constructed=[],
                   construction_is_not_SDK_delivery_or_attack_success=True)
    if not enabled:
        yield receipt
        return
    from policies.hybrid_skills_planner import DynamicFirstStaticPlanner
    original = DynamicFirstStaticPlanner.__init__
    if getattr(original, '_static_physical15_scope', False):
        raise RuntimeError('static_physical15_scope_already_installed')
    signature = inspect.signature(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original(self, *args, **kwargs)
        dynamic = self.dynamic
        self.static = make_static(bound.arguments['sim_cfg'], bound.arguments['call_budget'], enabled=True)
        if self.dynamic is not dynamic:
            raise AssertionError('dynamic_branch_must_not_change')
        receipt['constructed'].append(dict(static_class=type(self.static).__name__,
            dynamic_class=type(dynamic).__name__, call_budget=self.budget,
            fallback_initial_ego_OBB_margin_m=15., background_margin_m=self.static.builder.margin,
            skill_sha256=self.static.physical15_skill_sha256))
    initialize._static_physical15_scope = True
    DynamicFirstStaticPlanner.__init__ = initialize
    try:
        yield receipt
    finally:
        DynamicFirstStaticPlanner.__init__ = original
