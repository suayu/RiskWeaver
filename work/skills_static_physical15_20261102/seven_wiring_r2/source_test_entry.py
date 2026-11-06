"""入口参数、结果目录生命周期及恢复回归；base worker均为零模型测试脚本。"""
from pathlib import Path
import json
import sys
import pytest

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0, str(ROOT))
from work.skills_static_physical15_20261102 import run_worker as entry
from policies.hybrid_skills_planner import DynamicFirstStaticPlanner


def fixture(tmp_path, monkeypatch, fail=False, construct=False):
    root = tmp_path
    parent = root/'work/fake'
    parent.mkdir(parents=True)
    (root/'experiments').mkdir()
    base = parent/'run_worker.py'
    text = '''from pathlib import Path
import sys
from types import SimpleNamespace as NS
def main():
    out=Path(sys.argv[sys.argv.index('--output-dir')+1])
'''
    if construct:
        text += '''    from policies.hybrid_skills_planner import DynamicFirstStaticPlanner
    cfg=NS(ego_profile=NS(),llm=NS(skills=NS(enabled=True,max_repairs=1)))
    DynamicFirstStaticPlanner(NS(calls=0),cfg,4)
'''
    if fail:
        text += "    raise RuntimeError('fixture worker failed')\n"
    else:
        text += "    out.mkdir()\n    (out/'argv.json').write_text(__import__('json').dumps(sys.argv))\n    return 0\n"
    base.write_text(text)
    monkeypatch.setattr(entry, 'ROOT', root)
    return base, root/'experiments/arm'


def test_default_off_forwards_arguments_and_preserves_constructor(tmp_path, monkeypatch):
    base, out = fixture(tmp_path, monkeypatch)
    original, argv = DynamicFirstStaticPlanner.__init__, list(sys.argv)
    args = ['--base-worker', str(base), '--output-dir', str(out), '--skills', 'off', '--model', 'fixture']
    assert entry.main(args) == 0
    assert DynamicFirstStaticPlanner.__init__ is original and sys.argv == argv
    forwarded = json.loads((out/'argv.json').read_text())
    assert forwarded == [str(base), '--output-dir', str(out), '--skills', 'off', '--model', 'fixture']
    report = json.loads((out/'static_physical15_entry.json').read_text())
    assert report['requested_mode'] == 'off' and report['source_unchanged']
    assert report['scope']['constructed'] == []


def test_on_uses_factory_but_public_worker_creates_directory(tmp_path, monkeypatch):
    base, out = fixture(tmp_path, monkeypatch, construct=True)
    original = DynamicFirstStaticPlanner.__init__
    assert entry.main(['--base-worker', str(base), '--static-route-physical15', 'on',
                       '--output-dir', str(out), '--skills', 'on']) == 0
    assert DynamicFirstStaticPlanner.__init__ is original
    report = json.loads((out/'static_physical15_entry.json').read_text())
    assert report['scope']['constructed'][0]['fallback_initial_ego_OBB_margin_m'] == 15
    assert report['scope']['constructed'][0]['background_margin_m'] == .5


def test_existing_output_is_not_modified(tmp_path, monkeypatch):
    base, out = fixture(tmp_path, monkeypatch)
    out.mkdir()
    marker = out/'original.txt'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        entry.main(['--base-worker', str(base), '--output-dir', str(out)])
    assert list(out.iterdir()) == [marker]
    assert marker.read_text() == 'preserve'


def test_error_restores_and_does_not_precreate_output(tmp_path, monkeypatch):
    base, out = fixture(tmp_path, monkeypatch, fail=True)
    original, argv, path = DynamicFirstStaticPlanner.__init__, list(sys.argv), list(sys.path)
    with pytest.raises(RuntimeError, match='fixture worker failed'):
        entry.main(['--base-worker', str(base), '--static-route-physical15', 'on',
                    '--output-dir', str(out), '--skills', 'on'])
    assert DynamicFirstStaticPlanner.__init__ is original
    assert sys.argv == argv and sys.path == path and not out.exists()
    assert json.loads(out.with_name('arm_static_physical15_entry.json').read_text())['error_type'] == 'RuntimeError'


def test_on_cannot_silently_change_or_claim_legacy_comparison(tmp_path, monkeypatch):
    base, out = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='requires_skills_on'):
        entry.main(['--base-worker', str(base), '--static-route-physical15', 'on',
                    '--output-dir', str(out), '--skills', 'off'])
    assert not out.exists()


def test_base_worker_is_restricted_to_workspace(tmp_path, monkeypatch):
    _, out = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='independent_workspace'):
        entry.main(['--base-worker', str(tmp_path/'outside/run_worker.py'), '--output-dir', str(out)])
