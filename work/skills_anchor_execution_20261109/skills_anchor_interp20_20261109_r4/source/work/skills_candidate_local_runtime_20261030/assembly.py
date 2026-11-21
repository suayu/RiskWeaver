"""将已测高层配置装入真实时序工厂；退出恢复，不更换连续模型或求解器。"""
from contextlib import contextmanager
from pathlib import Path
import hashlib,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_temporal_candidates_20261029'),str(ROOT/'work/skills_identity_evidence_20261030')]
import temporal_planner
from common_evidence import install_common_evidence


@contextmanager
def install(grounding_mode='global',common_facts=False):
    """原global/无共同事实时不改变工厂；显式启用后通过配置覆盖保存真实有效值。"""
    if grounding_mode not in ('global','candidate_local') or not isinstance(common_facts,bool):raise ValueError('invalid_highlevel_assembly_flags')
    if grounding_mode=='global' and not common_facts:
        yield dict(enabled=False);return
    import run_closed_loop
    old_factory=temporal_planner.TemporalCandidatePlanner;old_install=run_closed_loop.install_variant
    expected=grounding_mode
    class ConfiguredTemporalPlanner(old_factory):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.grounding_validation_mode!=expected:raise ValueError('effective_highlevel_configuration_mismatch')
            if common_facts:install_common_evidence(self)
            self.highlevel_factory_assembly=dict(version='candidate_local_runtime_factory_20261030_v1',
                grounding_validation_mode=self.grounding_validation_mode,common_identity_comparison_facts=common_facts,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),diffusion_unchanged=True)
    def install_variant(options,runner):
        restore=old_install(options,runner);original=runner.group_overrides
        runner.group_overrides=lambda *args,**kwargs:original(*args,**kwargs)+['sim.llm.skills.grounding_validation_mode='+expected]
        def undo():runner.group_overrides=original;restore()
        return undo
    temporal_planner.TemporalCandidatePlanner=ConfiguredTemporalPlanner;run_closed_loop.install_variant=install_variant
    try:yield dict(enabled=True,grounding_validation_mode=expected,common_identity_comparison_facts=common_facts)
    finally:temporal_planner.TemporalCandidatePlanner=old_factory;run_closed_loop.install_variant=old_install
