"""45米、事件调度、动静态Skills的显式入口，沿用原资源和物理预算围栏。"""
from pathlib import Path
import importlib
import json
import runpy
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
INSTRUCTION=['优先选择可行且能与自车未来路径形成危险交互的车辆攻击。没有合适动态候选、动态不适合或明确不攻击时，选择合法静态物体组合拦截道路。仅选择代码发布的候选，不编造对象、锚点或可行性；保留原物理、道路和背景安全要求。预测不是实际发生的事件。']
OVERRIDES=['sim.planning_observation_radius_m=45.0','sim.llm.planning_trigger_mode=event',
    'sim.llm.skills.dynamic_first_in_plan=true','sim.llm.skills.static_fallback_enabled=true',
    'sim.llm.skills.source_fact_contract_enabled=true','sim.llm.attack_mode=joint',
    'sim.llm.obstacles.enabled=false','sim.rag.enabled=false']


def main():
    output=Path(sys.argv[sys.argv.index('--output-dir')+1]);prepare='--prepare-only' in sys.argv
    if output.exists():raise FileExistsError('preserve_existing_event_run')
    sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_v2_compact_dev_20261020')]
    module=importlib.import_module('run_closed_loop')
    if Path(module.__file__).resolve()!=ROOT/'work/skills_v2_compact_dev_20261020/run_closed_loop.py':
        raise ValueError('unexpected_base_runner')
    original=module.install_variant
    def variant(options,runner):
        previous=list(runner.INSTRUCTION)
        restore=original(options,runner);overrides=runner.group_overrides
        runner.group_overrides=lambda *a,**k:overrides(*a,**k)+OVERRIDES
        runner.INSTRUCTION=list(INSTRUCTION)
        def undo():
            runner.group_overrides=overrides;runner.INSTRUCTION=previous;restore()
        return undo
    module.install_variant=variant
    try:
        runpy.run_path(str(ROOT/'work/realtime_planner_20261104/run_worker.py'),run_name='__main__')
    finally:
        module.install_variant=original
        if not prepare and (output/'resolved_config.yaml').exists():
            from omegaconf import OmegaConf
            cfg=OmegaConf.load(output/'resolved_config.yaml')
            valid=(cfg.sim.planning_observation_radius_m==45 and cfg.sim.llm.planning_trigger_mode=='event'
                and cfg.sim.llm.skills.dynamic_first_in_plan and cfg.sim.llm.skills.static_fallback_enabled
                and cfg.sim.llm.attack_mode=='joint' and not cfg.sim.rag.enabled)
            with (output/'event_joint_entry.json').open('x',encoding='utf-8') as f:
                json.dump(dict(actual_config_verified=bool(valid),overrides=OVERRIDES,instruction=INSTRUCTION,
                               visualization_unchanged=True,native_FOV_unchanged=True),f,ensure_ascii=False,indent=2)
            if not valid:raise ValueError('event_joint_configuration_mismatch')


if __name__=='__main__':
    main()
