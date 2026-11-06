"""独立默认关闭工厂；不安装到公共代码，也不接管当前冻结实验。"""
import copy
import hashlib
from pathlib import Path

from facts import build_facts


def make_static_planner(sim_cfg, call_budget=4, enabled=False):
    """关时直接返回原类；开时在目录哈希计算前追加事实和固定技能。"""
    from policies.hybrid_skills_planner import StaticCombinationPlanner
    if not enabled:
        return StaticCombinationPlanner(sim_cfg, call_budget)

    class MetricFactStaticPlanner(StaticCombinationPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            original_load = self.registry.load

            def load(*args, **kwargs):
                knowledge = copy.deepcopy(original_load(*args, **kwargs))
                if self.registry.level in ('analysis', 'full'):
                    raw = Path(__file__).with_name('SKILL.md').read_bytes()
                    knowledge['documents'].append(dict(path='prototype/rw-static-metric-facts/SKILL.md',
                        sha256=hashlib.sha256(raw).hexdigest(), text=raw.decode('utf-8')))
                    knowledge['version'] += '+static-metric-facts-prototype-v1'
                return knowledge
            self.registry.load = load

        def build_context(self, *args, **kwargs):
            context = super().build_context(*args, **kwargs)
            context['static_metric_facts'] = build_facts(context['obstacle_candidates'])
            return context

    return MetricFactStaticPlanner(sim_cfg, call_budget)
