"""新版两组共用精确事实开关，保留旧RAG入口及全部原有模型/求解预算。"""
from pathlib import Path
import importlib.util

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
path = ROOT/'work/rag_integration_20261030/run_rag_worker.py'
spec = importlib.util.spec_from_file_location('preserved_rag_worker', path)
worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
original_overrides = worker.overrides


def common_overrides(enabled, inherit_path):
    return original_overrides(enabled, inherit_path) + [
        'sim.llm.skills.source_fact_contract_enabled=true']


if __name__ == '__main__':
    worker.overrides = common_overrides
    worker.main()
