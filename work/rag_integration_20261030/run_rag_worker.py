"""两组共用同一高层真实入口；只改变是否检索，均向本组新知识库记录事件。"""
from pathlib import Path
import argparse,json,sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def overrides(enabled,inherit_path):
    values=['sim.rag.enabled='+str(enabled).lower(),'sim.rag.record_events=true','sim.rag.knowledge_mode=new',
        'sim.rag.inherit='+str(inherit_path is not None).lower(),
        'sim.rag.inherit_path='+json.dumps(str(inherit_path)) if inherit_path else 'sim.rag.inherit_path=null']
    return values


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--rag',choices=('off','on'),required=True)
    p.add_argument('--rag-inherit',action='store_true');p.add_argument('--rag-source',type=Path)
    a,remaining=p.parse_known_args();output=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    if a.rag_inherit!=(a.rag_source is not None):raise ValueError('explicit_RAG_inherit_and_source_must_match')
    sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_temporal_candidates_20261029'),str(ROOT/'work/skills_candidate_local_runtime_20261030')]
    # 外层配置包装在所有既有工厂外生效；Generator在真实调用前绑定最终规划器。
    import assembly,run_closed_loop,run_worker
    original=run_closed_loop.install_variant;argv=sys.argv
    def install_variant(options,runner):
        restore=original(options,runner);old=runner.group_overrides
        runner.group_overrides=lambda *args,**kwargs:old(*args,**kwargs)+overrides(a.rag=='on',a.rag_source)
        def undo():runner.group_overrides=old;restore()
        return undo
    run_closed_loop.install_variant=install_variant
    try:
        sys.argv=[sys.argv[0],*remaining];return run_worker.main()
    finally:
        sys.argv=argv;run_closed_loop.install_variant=original
        if not prepare and (output/'resolved_config.yaml').exists():
            from omegaconf import OmegaConf
            cfg=OmegaConf.load(output/'resolved_config.yaml');settings=cfg.sim.rag
            verified=bool(settings.enabled)==(a.rag=='on') and bool(settings.record_events) and bool(settings.runtime_knowledge_path)
            with (output/'rag_entry_protocol.json').open('x',encoding='utf-8') as file:
                json.dump(dict(retrieval_enabled=bool(settings.enabled),record_events=bool(settings.record_events),
                    knowledge_path=settings.runtime_knowledge_path,experiment_id=settings.runtime_experiment_id,
                    effective_config_verified=verified,legacy_history_direct_prompt_disabled=True),file,indent=2)
            if not verified:raise RuntimeError('actual_RAG_entry_configuration_not_applied')


if __name__=='__main__':main()
