"""独立后续入口：保留v1配置/传输装配，只补充共同混合攻击指令。"""
import importlib
import importlib.util
import sys
from pathlib import Path

from instruction_scope import install

ROOT = Path('/home2/zhaoyx/scenario-dreamer')


def main():
    sys.path[:0] = [str(ROOT), str(ROOT/'work/hybrid_closed_loop_20261102'),
        str(ROOT/'work/skills_temporal_candidates_20261029'),
        str(ROOT/'work/skills_candidate_local_runtime_20261030'),
        str(ROOT/'work/skills_v2_compact_dev_20261020')]
    module = importlib.import_module('run_closed_loop')
    if Path(module.__file__).resolve() != ROOT/'work/skills_v2_compact_dev_20261020/run_closed_loop.py':
        raise ValueError('unexpected_closed_loop_assembly_module')
    output = Path(sys.argv[sys.argv.index('--output-dir')+1])
    prepare = '--prepare-only' in sys.argv
    spec = importlib.util.spec_from_file_location('preserved_hybrid_worker',
        ROOT/'work/hybrid_closed_loop_20261102/run_rag_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    with install(module, output=None if prepare else output):
        return worker.main()


if __name__ == '__main__':
    main()
