"""独立CLI启用混合路由预览一致性；模型/预算仍由显式选择的原工作入口控制。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from work.skills_hybrid_preview_20261102.runtime import install

HERE = Path(__file__).resolve().parent
CORE = ['runtime.py', 'run_worker.py']


def source_hashes():
    return {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in CORE}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--base-worker', type=Path, required=True)
    parser.add_argument('--consistent-routing-preview', choices=('off', 'on'), default='off')
    options, remaining = parser.parse_known_args(argv)
    base = options.base_worker.resolve()
    if not base.is_relative_to(ROOT/'work') or base.name != 'run_worker.py' or base == Path(__file__).resolve():
        raise ValueError('base_worker_must_be_an_independent_workspace_run_worker')
    if '--output-dir' not in remaining:
        raise ValueError('output_dir_required')
    output = Path(remaining[remaining.index('--output-dir')+1])
    if output.exists():
        raise FileExistsError('preserve_existing_output_before_loading_any_worker')
    enabled = options.consistent_routing_preview == 'on'
    if enabled:
        for flag in ('--skills', '--rolling-intent', '--fractional-interaction'):
            if flag not in remaining or remaining[remaining.index(flag)+1] != 'on':
                raise ValueError('consistent_preview_requires_on:'+flag)
    prepare = '--prepare-only' in remaining
    before = source_hashes()
    old_argv, old_path = list(sys.argv), list(sys.path)
    scope = dict(enabled=enabled, calls=[])
    error = None
    try:
        # 与直接执行该脚本相同的模块搜索顺序，避免其transport被其他私有入口误绑定。
        sys.path.insert(0, str(base.parent))
        spec = importlib.util.spec_from_file_location('preserved_physical15_base_worker', base)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        sys.argv = [str(base), *remaining]
        with install(enabled) as scope:
            return worker.main()
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        sys.argv, sys.path[:] = old_argv, old_path
        receipt = dict(version='consistent_routing_preview_entry_v1', base_worker=str(base),
            base_worker_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
            requested_mode=options.consistent_routing_preview, prepare_only=prepare,
            scope=scope, source_sha256=before, source_unchanged=source_hashes() == before,
            error_type=error, actual_SDK_delivery_and_threat_still_require_verification=True)
        if not prepare and output.parent.is_dir():
            # 输出目录由原公共runner创建；本层不提前创建、不覆盖原目录。
            target = (output/'consistent_routing_preview_entry.json' if output.is_dir()
                      else output.with_name(output.name+'_consistent_routing_preview_entry.json'))
            with target.open('x', encoding='utf-8') as file:
                json.dump(receipt, file, ensure_ascii=False, indent=2)
        if not receipt['source_unchanged'] and error is None:
            raise RuntimeError('consistent_routing_preview_source_changed_during_worker')


if __name__ == '__main__':
    main()
