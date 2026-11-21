"""独立开关入口；关闭时完全复用v4，开启时仅替换高级规划器与知识根目录。"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'work/skills_recovery_v4_20261023'),
               str(ROOT/'work/skills_v2_compact_dev_20261020')]
import run_recovery_v4 as base


def priority_override(install, options, runner):
    restore = install(options, runner)
    previous = runner.group_overrides
    def overrides(*args, **kwargs):
        # 最后一个root与实际加载根一致，避免resolved_config仍记录旧知识包。
        return previous(*args, **kwargs) + ['sim.llm.skills.root='+str(HERE/'skills')]
    runner.group_overrides = overrides
    def undo():
        runner.group_overrides = previous
        restore()
    return undo


def run(options, remaining):
    if options.priority_skills == 'off':
        return base.main()
    for flag in ['--skills','--compact-view','--event-budget','--reachability-preflight','--route-tail-preflight']:
        if remaining.count(flag) != 1 or remaining[remaining.index(flag)+1] != 'on':
            raise ValueError('priority_requires_explicit_pipeline:' + flag)
    files = [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.json'}]
    hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    receipt = dict(priority_skills='on', version='priority-v5-20261024', knowledge_root=str(HERE/'skills'),
        bailian_capabilities=options.bailian_capabilities,
        priorities=['correct_scene_and_attack','actual_ego_threat','enabled_NL_control','enabled_difficulty_control'],
        source_sha256=hashes, continuous_refinement_changed=False,
        scope='dynamic planning first; optional controls and full correctness/actual threat still require experiments')
    if '--prepare-only' in remaining:
        stream = io.StringIO()
        with redirect_stdout(stream):base.main()
        result = json.loads(stream.getvalue());result['priority_protocol'] = receipt
        print(json.dumps(result,ensure_ascii=False));return
    output = Path(remaining[remaining.index('--output-dir')+1])
    frozen = json.loads((output.parent/'source_sha256.json').read_text())
    if any(frozen.get(n) != d for n,d in hashes.items()):
        raise ValueError('priority_sources_not_frozen_or_changed')
    from priority_planner import PrioritySkillsPlanner
    import run_closed_loop
    previous_planner = base.RouteTailReadyPlanner
    previous_install = run_closed_loop.install_variant
    base.RouteTailReadyPlanner = PrioritySkillsPlanner
    run_closed_loop.install_variant = lambda options,runner:priority_override(previous_install,options,runner)
    try:
        base.main()
    finally:
        run_closed_loop.install_variant = previous_install
        base.RouteTailReadyPlanner = previous_planner
        if output.exists():
            with (output/'priority_protocol.json').open('x',encoding='utf-8') as file:
                json.dump(receipt,file,ensure_ascii=False,indent=2)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--priority-skills', choices=('off','on'), default='off')
    parser.add_argument('--bailian-capabilities', choices=('legacy','observed'), default='legacy')
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    restore = lambda:None
    if options.bailian_capabilities == 'observed':
        from bailian_transport import install_provider_compatibility
        restore = install_provider_compatibility()
    try:
        return run(options, remaining)
    finally:
        restore()
        if options.bailian_capabilities == 'observed' and '--prepare-only' not in remaining and '--output-dir' in remaining:
            output = Path(remaining[remaining.index('--output-dir')+1])
            if output.exists():
                adapter = HERE/'bailian_transport.py'
                with (output/'bailian_capabilities_protocol.json').open('x',encoding='utf-8') as file:
                    json.dump(dict(mode='observed',common_to_comparison_arms=True,
                        exact_model='glm-5.3',enable_thinking=True,additional_requests=0,
                        source_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest()),file,indent=2)


if __name__ == '__main__':main()
