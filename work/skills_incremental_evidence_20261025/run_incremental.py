"""独立增量技能入口；默认关闭，真实运行须新冻结源码，不改已有任务。"""
import argparse
from contextlib import redirect_stdout
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'work/skills_priority_v5_20261024'), str(ROOT/'work/skills_recovery_v4_20261023')]
import run_priority as previous


def install_root(install, options, runner):
    restore = install(options, runner)
    original = runner.group_overrides
    runner.group_overrides = lambda *args, **kwargs:original(*args, **kwargs)+['sim.llm.skills.root='+str(HERE/'skills')]
    def undo():
        runner.group_overrides = original
        restore()
    return undo


def run(options, remaining):
    if options.incremental_skills == 'off':
        if options.active_window_recipes != 'off':
            raise ValueError('new_recipes_require_explicit_incremental_switch')
        # 旧调用路径自行解析原参数，不强制启用skills或新的provider兼容。
        return previous.main()
    for flag in ['--skills','--compact-view','--event-budget','--reachability-preflight','--route-tail-preflight']:
        if remaining.count(flag) != 1 or remaining[remaining.index(flag)+1] != 'on':
            raise ValueError('incremental_requires_explicit_pipeline:'+flag)
    if options.active_window_recipes == 'on' and ('--candidate-expansion' not in remaining
            or remaining[remaining.index('--candidate-expansion')+1] != 'on'):
        raise ValueError('active_window_recipes_require_candidate_expansion')
    if '--prepare-only' not in remaining and Path(remaining[remaining.index('--output-dir')+1]).exists():
        raise FileExistsError('preserve_existing_incremental_experiment')
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
              for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.json'}}
    receipt = dict(version='incremental-prototype-20261025', enabled=True,
        extra_recipes=options.active_window_recipes == 'on', knowledge_root=str(HERE/'skills'), source_sha256=hashes,
        bailian_capabilities=options.bailian_capabilities,
        actual_threat_validated=False, continuous_refinement_changed=False,
        preserves_current_and_route_tail_preflight=True, previous_v5_unchanged=True)
    if '--prepare-only' in remaining:
        stream = io.StringIO()
        with redirect_stdout(stream):previous.base.main()
        data = json.loads(stream.getvalue());data['incremental_protocol'] = receipt
        print(json.dumps(data, ensure_ascii=False))
        return
    output = Path(remaining[remaining.index('--output-dir')+1])
    frozen = json.loads((output.parent/'source_sha256.json').read_text())
    if any(frozen.get(name) != digest for name, digest in hashes.items()):
        raise ValueError('new_incremental_sources_not_frozen_or_changed')
    from incremental_planner import IncrementalSkillsPlanner
    import run_closed_loop
    original_planner = previous.base.RouteTailReadyPlanner
    original_install = run_closed_loop.install_variant
    previous.base.RouteTailReadyPlanner = partial(IncrementalSkillsPlanner, extra_recipes=receipt['extra_recipes'])
    run_closed_loop.install_variant = lambda options, runner:install_root(original_install, options, runner)
    try:
        previous.base.main()
    finally:
        previous.base.RouteTailReadyPlanner = original_planner
        run_closed_loop.install_variant = original_install
        if output.exists():
            with (output/'incremental_protocol.json').open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--incremental-skills', choices=('off','on'), default='off')
    parser.add_argument('--active-window-recipes', choices=('off','on'), default='off')
    parser.add_argument('--bailian-capabilities', choices=('legacy','observed'), default='legacy')
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]]+remaining
    # 在进入任何finally产物写入前拒绝旧目录，失败重跑不能向历史实验加收据。
    if '--prepare-only' not in remaining and '--output-dir' in remaining:
        if Path(remaining[remaining.index('--output-dir')+1]).exists():
            raise FileExistsError('preserve_existing_incremental_experiment')
    # 新/旧比较可共同打开同一provider兼容；不会产生额外请求。
    restore = lambda:None
    if options.bailian_capabilities == 'observed':
        from bailian_transport import install_provider_compatibility
        restore = install_provider_compatibility()
    try:
        return run(options, remaining)
    finally:
        restore()
        if '--prepare-only' not in remaining and '--output-dir' in remaining:
            output = Path(remaining[remaining.index('--output-dir')+1])
            if output.exists():
                with (output/'incremental_entry_protocol.json').open('x', encoding='utf-8') as stream:
                    json.dump(dict(incremental_skills=options.incremental_skills,
                        active_window_recipes=options.active_window_recipes, bailian_capabilities=options.bailian_capabilities,
                        additional_provider_requests=0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)


if __name__ == '__main__':main()
