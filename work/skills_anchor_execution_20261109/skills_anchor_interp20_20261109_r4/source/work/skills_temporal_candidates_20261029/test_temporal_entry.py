"""入口开关隔离和回滚，不创建场景、不使用真实API。"""
from functools import partial
import sys
import pytest
import run_temporal_controlled as entry


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_entry_factory_arguments_and_restoration(tmp_path,monkeypatch,enabled):
    sys.path.insert(0,str(entry.ROOT/'work/skills_rolling_contract_20261029'))
    import contract_planner,run_contract_controlled
    old=contract_planner.ContractRollingPlanner
    argv=['run','--fractional-interaction','on' if enabled else 'off','--velocity-continuation','on' if enabled else 'off',
        '--motion-evidence','on' if enabled else 'off','--skills','on','--incremental-skills','on',
        '--rolling-intent','on','--explicit-contract','on','--output-dir',str(tmp_path/'new'),'--prepare-only']
    monkeypatch.setattr(sys,'argv',argv)
    def fake():
        current=contract_planner.ContractRollingPlanner
        if enabled:assert isinstance(current,partial) and current.keywords==dict(fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
        else:assert current is old
        assert '--fractional-interaction' not in sys.argv
        return 'same_parent'
    monkeypatch.setattr(run_contract_controlled,'main',fake)
    assert entry.main()=='same_parent' and contract_planner.ContractRollingPlanner is old and sys.argv is argv
    assert not (tmp_path/'new').exists()


def test_incompatible_old_path_refused_instead_of_silently_ignoring_switch(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['run','--fractional-interaction','on','--skills','off','--output-dir',str(tmp_path/'new')])
    with pytest.raises(ValueError,match='requires_explicit'):entry.main()
    assert not (tmp_path/'new').exists()
