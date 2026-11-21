"""真实入口参数和工厂恢复回归；零API、零GPU。"""
from functools import partial
import sys
import pytest
import run_rolling_controlled as entry


@pytest.mark.parametrize('enabled',[False,True])
def test_factory_and_arguments_restored(tmp_path,monkeypatch,enabled):
    sys.path[:0]=[str(entry.ROOT/'work/skills_shared_resource_20261028'),str(entry.ROOT/'work/skills_incremental_evidence_20261025')]
    import run_capped,incremental_planner
    original=incremental_planner.IncrementalSkillsPlanner
    argv=['run','--rolling-intent','on' if enabled else 'off','--skills','on','--incremental-skills','on',
        '--active-window-recipes','on','--output-dir',str(tmp_path/'new'),'--prepare-only']
    monkeypatch.setattr(sys,'argv',argv)
    def fake():
        factory=incremental_planner.IncrementalSkillsPlanner
        if enabled:assert isinstance(factory,partial) and factory.keywords==dict(rolling_intent=True)
        else:assert factory is original
        assert '--rolling-intent' not in sys.argv
        return 'actual_parent'
    monkeypatch.setattr(run_capped,'main',fake)
    assert entry.main()=='actual_parent' and incremental_planner.IncrementalSkillsPlanner is original and sys.argv is argv
    assert not (tmp_path/'new').exists()


def test_wrong_switch_combination_cannot_start(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['run','--rolling-intent','on','--skills','off','--output-dir',str(tmp_path/'new')])
    with pytest.raises(ValueError,match='requires_full'):entry.main()
    assert not (tmp_path/'new').exists()


def test_existing_output_preserved(tmp_path,monkeypatch):
    old=tmp_path/'old';old.mkdir();(old/'marker').write_text('original')
    monkeypatch.setattr(sys,'argv',['run','--output-dir',str(old)])
    with pytest.raises(FileExistsError,match='preserve'):entry.main()
    assert (old/'marker').read_text()=='original'
