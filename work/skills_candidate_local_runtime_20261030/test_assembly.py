"""真实工厂、配置覆盖、严格模式和异常恢复；不联网、不初始化GPU。"""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(ROOT/'work/skills_temporal_candidates_20261029')]
import assembly
from replay_074 import source_fixture


def test_disabled_assembly_keeps_original_factory():
    original=assembly.temporal_planner.TemporalCandidatePlanner
    with assembly.install():assert assembly.temporal_planner.TemporalCandidatePlanner is original


def test_real_factory_effective_configuration_and_common_prompt_layer():
    env,*_=source_fixture();cfg=OmegaConf.create(OmegaConf.to_container(env.cfg,resolve=True));cfg.sim.llm.skills.grounding_validation_mode='candidate_local'
    original=assembly.temporal_planner.TemporalCandidatePlanner
    with assembly.install('candidate_local',True):
        engine=assembly.temporal_planner.TemporalCandidatePlanner(cfg.sim,4,extra_recipes=True,rolling_intent=True,
            explicit_contract=True,deduplicate=True,fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
        assert engine.grounding_validation_mode=='candidate_local' and 'plan' in engine.__dict__
        assert engine.highlevel_factory_assembly['common_identity_comparison_facts']
    assert assembly.temporal_planner.TemporalCandidatePlanner is original


def test_resolved_group_configuration_records_effective_policy_and_exception_restores(monkeypatch):
    import run_closed_loop
    def original_install(options,runner):return lambda:None
    monkeypatch.setattr(run_closed_loop,'install_variant',original_install)
    runner=NS(group_overrides=lambda *a,**k:['original=unchanged'])
    original=assembly.temporal_planner.TemporalCandidatePlanner
    with pytest.raises(RuntimeError,match='worker_failure'):
        with assembly.install('candidate_local',True):
            undo=run_closed_loop.install_variant(None,runner)
            assert runner.group_overrides()==['original=unchanged','sim.llm.skills.grounding_validation_mode=candidate_local']
            undo();assert runner.group_overrides()==['original=unchanged']
            raise RuntimeError('worker_failure')
    assert run_closed_loop.install_variant is original_install and assembly.temporal_planner.TemporalCandidatePlanner is original


def test_factory_cannot_claim_local_if_runtime_config_says_global():
    env,*_=source_fixture()
    with assembly.install('candidate_local',True):
        with pytest.raises(ValueError,match='configuration_mismatch'):
            assembly.temporal_planner.TemporalCandidatePlanner(env.cfg.sim,4,extra_recipes=True,rolling_intent=True,
                explicit_contract=True,deduplicate=True,fractional_interaction=True,velocity_continuation=True,motion_evidence=True)


def test_actual_nested_entry_reaches_configured_factory_and_restores_all_wrappers(tmp_path,monkeypatch):
    # 穿过真实temporal/contract/rolling入口，在资源/仿真入口边界停止，不启动CUDA。
    sys.path.insert(0,str(ROOT/'work/skills_shared_resource_20261028'))
    import run_capped,run_worker
    env,*_=source_fixture();cfg=OmegaConf.create(OmegaConf.to_container(env.cfg,resolve=True))
    cfg.sim.llm.skills.grounding_validation_mode='candidate_local'
    original=assembly.temporal_planner.TemporalCandidatePlanner
    def final_resource_boundary():
        import incremental_planner
        return incremental_planner.IncrementalSkillsPlanner(cfg.sim,4,extra_recipes=True)
    monkeypatch.setattr(run_capped,'main',final_resource_boundary)
    args=['entry','--highlevel-grounding','candidate_local','--identity-comparison-facts','on','--output-dir',str(tmp_path/'unused'),'--prepare-only']
    for flag in ['skills','incremental-skills','active-window-recipes','rolling-intent','explicit-contract','deduplicate-intents','fractional-interaction','velocity-continuation','motion-evidence']:
        args += ['--'+flag,'on']
    monkeypatch.setattr(sys,'argv',args)
    engine=run_worker.main()
    assert engine.grounding_validation_mode=='candidate_local' and engine.motion_evidence and engine.fractional_interaction
    assert engine.explicit_contract and engine.rolling_intent and engine.highlevel_factory_assembly['common_identity_comparison_facts']
    assert assembly.temporal_planner.TemporalCandidatePlanner is original and not (tmp_path/'unused').exists()
