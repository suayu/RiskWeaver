"""显式参数开启动态优先/静态兜底；默认关闭，复用既有skills与RAG实际入口。"""
from pathlib import Path
import argparse,importlib.util,json,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--static-fallback',choices=('off','on'),default='off')
    parser.add_argument('--source-fact-contract',choices=('off','on'),default='off')
    args,remaining=parser.parse_known_args()
    output=Path(remaining[remaining.index('--output-dir')+1]);prepare='--prepare-only' in remaining
    path=ROOT/'work/rag_integration_20261030/run_rag_worker.py'
    spec=importlib.util.spec_from_file_location('preserved_hybrid_base',path)
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    overrides=worker.overrides
    if args.source_fact_contract=='on':
        worker.overrides=lambda *a,**k:overrides(*a,**k)+['sim.llm.skills.source_fact_contract_enabled=true']
    from hybrid_assembly import install as assembly
    from hybrid_rag import install as rag
    previous=sys.argv
    try:
        sys.argv=[sys.argv[0],*remaining]
        if args.static_fallback=='on':
            with rag(),assembly():return worker.main()
        return worker.main()
    finally:
        sys.argv=previous
        if not prepare and (output/'resolved_config.yaml').exists():
            from omegaconf import OmegaConf
            cfg=OmegaConf.load(output/'resolved_config.yaml')
            (output/'hybrid_entry_protocol.json').write_text(json.dumps(dict(
                static_fallback_enabled=bool(getattr(cfg.sim.llm.skills,'static_fallback_enabled',False)),
                attack_mode=cfg.sim.llm.attack_mode,source_fact_contract_enabled=cfg.sim.llm.skills.source_fact_contract_enabled,
                static_and_dynamic_counted_separately=True,actual_execution_not_certified_by_plan=True),indent=2))


if __name__=='__main__':main()
