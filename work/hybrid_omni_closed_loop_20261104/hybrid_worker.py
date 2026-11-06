"""公共动态/静态Skills真实闭环；两组只改变RAG，冷启动知识库按原生命周期记账。"""
from pathlib import Path
import importlib.util,json,sys
from transport import install

ROOT=Path('/home2/zhaoyx/scenario-dreamer')


def common_overrides(original,*args,**kwargs):
    return original(*args,**kwargs)+['sim.llm.skills.static_fallback_enabled=true',
        'sim.llm.skills.source_fact_contract_enabled=true','sim.llm.attack_mode=joint',
        'sim.llm.obstacles.enabled=false']


def main():
    output=Path(sys.argv[sys.argv.index('--output-dir')+1]);prepare='--prepare-only' in sys.argv
    spec=importlib.util.spec_from_file_location('preserved_rag_worker',ROOT/'work/rag_integration_20261030/run_rag_worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    original=worker.overrides;worker.overrides=lambda *a,**k:common_overrides(original,*a,**k)
    try:
        with install(output):return worker.main()
    finally:
        if not prepare and (output/'resolved_config.yaml').exists():
            from omegaconf import OmegaConf
            cfg=OmegaConf.load(output/'resolved_config.yaml')
            verified=(cfg.sim.llm.skills.static_fallback_enabled and cfg.sim.llm.skills.source_fact_contract_enabled
                and cfg.sim.llm.attack_mode=='joint' and not cfg.sim.llm.obstacles.enabled)
            receipt=dict(version='hybrid_closed_loop_omni_v1_20261104',actual_configuration_verified=bool(verified),
                timeout_seconds=180,max_tokens=8192,reasoning_effort=None,physical_calls_per_scene=4,
                solver_and_safety_unchanged=True,diffusion_execution=True,new_proximal_correction_enabled=False)
            with (output/'hybrid_closed_loop_entry.json').open('x',encoding='utf-8') as file:json.dump(receipt,file,indent=2)
            if not verified:raise RuntimeError('actual_hybrid_configuration_not_applied')


if __name__=='__main__':main()
