"""以显式开关装配公共传输修复；旧运行入口和默认调用机制保持可回退。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--semantic-model-policy', choices=('legacy', 'service_only'), default='legacy')
    options, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index('--output-dir')+1])
    prepare = '--prepare-only' in remaining
    if output.exists():
        raise FileExistsError('preserve_prior_experiment')
    # 前置模块导入不发送请求；实际预算、SDK审计和安全仍由原真实入口安装。
    sys.path[:0] = [str(ROOT), str(ROOT/'work/skills_incremental_evidence_20261025')]
    from provider_control import install_control, THINKING_REQUIRED
    import run_incremental
    if not prepare:
        hashes = json.loads((output.parent/'source_sha256.json').read_text())
        for path in [Path(__file__), HERE/'provider_control.py']:
            if hashes.get(path.relative_to(ROOT).as_posix()) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('provider_control_not_frozen_or_changed')
    restore = install_control(options.semantic_model_policy)
    old_argv = sys.argv
    try:
        sys.argv = [sys.argv[0]]+remaining
        return run_incremental.main()
    finally:
        sys.argv = old_argv
        restore()
        if not prepare and output.exists():
            receipt = dict(version='common_provider_control_20261028_v1',
                thinking_required_exact_models=sorted(THINKING_REQUIRED),
                semantic_model_policy=options.semantic_model_policy,
                true_service_fallback_unchanged=True, permanent_disable_only_404_or_explicit_quota=True,
                extra_requests=0, transport_limit_unchanged=True,
                skills_and_diffusion_and_safety_unchanged=True)
            with (output/'common_provider_control.json').open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
