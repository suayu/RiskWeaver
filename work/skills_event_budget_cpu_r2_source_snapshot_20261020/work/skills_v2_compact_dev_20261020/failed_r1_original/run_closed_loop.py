"""独立开发入口：可选压缩视图和真实API预算，不改冻结v1源文件。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0, str(ROOT))


def install_variant(options, runner):
    """仅替换本进程的生成器与配置函数，返回恢复函数。"""
    from compact_planner import CompactSkillsPlanner
    from event_budget_generator import EventBudgetGenerator
    original_generator = runner.AdversarialScenarioGenerator
    original_overrides = runner.group_overrides

    class DevelopmentGenerator(EventBudgetGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if options.compact_view == 'on':
                if self.skills_planner is None:
                    raise ValueError('compact view requires explicit skills enabled')
                self.skills_planner = CompactSkillsPlanner(self.cfg.sim, self._max_call_times)

    def overrides(*args, **kwargs):
        values = original_overrides(*args, **kwargs)
        values += [f'+sim.llm.skills.event_budget_enabled={str(options.event_budget == "on").lower()}',
                   f'+sim.llm.skills.max_screening_queries={options.max_screening_queries}']
        if options.device == 'cpu':
            # GPU正在运行冻结矩阵；此分支明确使用CPU，不把延迟与GPU推理混比。
            values += ['sim.traffic_model.device=cpu']
        if options.compact_view == 'on':
            values += [f'sim.llm.skills.root={Path(__file__).resolve().parent / "skills"}']
        return values

    runner.AdversarialScenarioGenerator = DevelopmentGenerator
    runner.group_overrides = overrides

    def restore():
        runner.AdversarialScenarioGenerator = original_generator
        runner.group_overrides = original_overrides
    return restore


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--compact-view', choices=('off', 'on'), default='off')
    parser.add_argument('--event-budget', choices=('off', 'on'), default='off')
    parser.add_argument('--max-screening-queries', type=int, default=40)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    options, remaining = parser.parse_known_args()
    if not 1 <= options.max_screening_queries <= 80:
        raise ValueError('screening budget must be within [1,80]')
    if options.device == 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU run requires CUDA_VISIBLE_DEVICES empty before process start')
        import torch
        torch.set_num_threads(2)
    from experiments.comparison import run_e1_group as runner
    from experiments.skills.run_instrumented import main as instrumented_main
    output = Path(remaining[remaining.index('--output-dir') + 1])
    if output.exists():
        raise FileExistsError(output)
    files = [Path(__file__).resolve(), Path(__file__).with_name('compact_planner.py'),
             Path(__file__).with_name('event_budget_generator.py')]
    frozen = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    restore = install_variant(options, runner)
    sys.argv = [sys.argv[0]] + remaining
    try:
        instrumented_main()
    finally:
        restore()
        if output.exists():
            protocol = dict(version='skills-v2-compact-event-development',
                            compact_view=options.compact_view, event_budget=options.event_budget,
                            max_screening_queries=options.max_screening_queries,
                            device=options.device, source_sha256=frozen,
                            source_unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h
                                                 for n, h in frozen.items()),
                            controls='same compiled catalog, solver, safety, 10-frame validity and 4-call budget; new screening budget separate',
                            scope='CPU development result; not merged into GPU v1 matrix')
            (output/'development_variant.json').write_text(
                json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
