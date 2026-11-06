"""实际公共runner目录检查回归；在模型/配置加载前用标记异常停止。"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from instruction_scope import INSTRUCTION, install

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0] = [str(ROOT), str(ROOT / 'work/skills_v2_compact_dev_20261020')]


def test_actual_runner_owns_output_creation(tmp_path, monkeypatch):
    module = importlib.import_module('run_closed_loop')
    from experiments.comparison import run_e1_group as runner
    before = list(runner.INSTRUCTION)
    out = tmp_path / 'arm'
    options = NS(compact_view='on', event_budget='on', reachability_preflight='on',
                 max_screening_queries=40, device='cpu')
    class StopBeforeConfiguration(Exception):
        pass
    with install(module, output=out):
        undo = module.install_variant(options, runner)
        assert not out.exists()
        assert runner.INSTRUCTION == INSTRUCTION
        with monkeypatch.context() as patch:
            patch.setattr(runner, 'validate_manifest', lambda _: ({'comparison_scene_count': 2}, []))
            def stop(*args, **kwargs):
                raise StopBeforeConfiguration()
            patch.setattr(runner, 'group_overrides', stop)
            patch.setattr(sys, 'argv', ['fixture', '--group', 'riskweaver_profile_off',
                '--manifest', str(tmp_path / 'unused.json'), '--output-dir', str(out)])
            with pytest.raises(StopBeforeConfiguration):
                runner.main()
        assert out.is_dir() and (out / 'manifest.json').exists()
        undo()
    assert runner.INSTRUCTION == before
    assert (out / 'hybrid_instruction_scope.json').exists()


def test_failed_before_public_directory_does_not_create_it(tmp_path):
    runner = NS(INSTRUCTION=['old'])
    module = NS(install_variant=lambda *args: lambda: None)
    out = tmp_path / 'arm'
    with install(module, output=out):
        module.install_variant(NS(), runner)
    assert not out.exists()
    assert (tmp_path / 'arm_hybrid_instruction_scope.json').exists()
    assert runner.INSTRUCTION == ['old']
