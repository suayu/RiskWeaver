"""入口默认兼容、工厂恢复及保护历史目录。"""
from functools import partial
import sys
import pytest
import run_contract_controlled as entry


@pytest.mark.parametrize('enabled',[False,True])
def test_factory_restore_and_switch_removal(tmp_path,monkeypatch,enabled):
    sys.path.insert(0,str(entry.ROOT/'work/skills_rolling_intent_20261028'))
    import rolling_planner,run_rolling_controlled
    original=rolling_planner.RollingIntentPlanner
    argv=['run','--explicit-contract','on' if enabled else 'off','--deduplicate-intents','on' if enabled else 'off',
        '--skills','on','--incremental-skills','on','--active-window-recipes','on','--rolling-intent','on',
        '--output-dir',str(tmp_path/'new'),'--prepare-only'];monkeypatch.setattr(sys,'argv',argv)
    def fake():
        value=rolling_planner.RollingIntentPlanner
        if enabled:assert isinstance(value,partial) and value.keywords==dict(explicit_contract=True,deduplicate=True)
        else:assert value is original
        assert '--explicit-contract' not in sys.argv and '--deduplicate-intents' not in sys.argv
        return 'same_parent'
    monkeypatch.setattr(run_rolling_controlled,'main',fake)
    assert entry.main()=='same_parent' and rolling_planner.RollingIntentPlanner is original and sys.argv is argv


def test_existing_directory_keeps_original_marker(tmp_path,monkeypatch):
    p=tmp_path/'old';p.mkdir();(p/'marker').write_text('old')
    monkeypatch.setattr(sys,'argv',['run','--output-dir',str(p)])
    with pytest.raises(FileExistsError,match='preserve'):entry.main()
    assert (p/'marker').read_text()=='old'
