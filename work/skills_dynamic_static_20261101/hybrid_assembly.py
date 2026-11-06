"""最终生成器完成既有时序工厂装配后再接静态兜底；关闭时完全沿用旧入口。"""
from contextlib import contextmanager
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_v2_compact_dev_20261020')]
from hybrid_planner import DynamicFirstStaticPlanner


@contextmanager
def install(enabled=True):
    if not enabled:
        yield;return
    from experiments.skills import run_instrumented
    original=run_instrumented.main
    def main(*args,**kwargs):
        # 此时所有旧时序工厂已经装配；再包装最终生成器，避免被后续子类替换覆盖。
        from experiments.comparison import run_e1_group as runner
        factory=runner.AdversarialScenarioGenerator
        overrides=runner.group_overrides
        class HybridGenerator(factory):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                if not bool(getattr(self.cfg.sim.llm.skills,'static_fallback_enabled',False)):
                    raise ValueError('hybrid_entry_configuration_not_effective')
                if self.skills_planner is None:raise ValueError('hybrid_entry_requires_skills')
                self.skills_planner=DynamicFirstStaticPlanner(self.skills_planner,self.cfg.sim,self._max_call_times)
                self.llm_planner.use_obstacles=True
        runner.AdversarialScenarioGenerator=HybridGenerator
        # 攻击模式控制整体静态能力；旧时序构造器的兼容布尔值保持false，动态子阶段按原契约运行。
        runner.group_overrides=lambda *a,**k:overrides(*a,**k)+[
            '++sim.llm.skills.static_fallback_enabled=true',
            'sim.llm.attack_mode=joint','sim.llm.obstacles.enabled=false']
        try:return original(*args,**kwargs)
        finally:
            runner.AdversarialScenarioGenerator=factory;runner.group_overrides=overrides
    run_instrumented.main=main
    try:yield
    finally:run_instrumented.main=original
