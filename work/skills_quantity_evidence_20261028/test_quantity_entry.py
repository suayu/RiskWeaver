"""验证真实入口临时装配、关闭兼容和保留既有目录；不创建新仿真。"""
from functools import partial
import sys
import pytest
import run_quantity_controlled as q


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_entry_factory_and_argv_restored(tmp_path,monkeypatch,enabled):
    sys.path[:0]=[str(q.ROOT/'work/skills_shared_resource_20261028'),str(q.ROOT/'work/skills_incremental_evidence_20261025')]
    import run_capped
    import incremental_planner
    original=incremental_planner.IncrementalSkillsPlanner;seen=[]
    argv=['run','--quantity-evidence','on' if enabled else 'off','--skills','on','--incremental-skills','on',
        '--output-dir',str(tmp_path/'output'),'--prepare-only']
    monkeypatch.setattr(sys,'argv',argv)
    def fake():
        factory=incremental_planner.IncrementalSkillsPlanner
        seen.append(factory)
        if enabled:assert isinstance(factory,partial) and factory.keywords==dict(quantity_evidence=True)
        else:assert factory is original
        assert '--quantity-evidence' not in sys.argv
        return 'same_actual_parent_path'
    monkeypatch.setattr(run_capped,'main',fake)
    assert q.main()=='same_actual_parent_path'
    assert len(seen)==1 and incremental_planner.IncrementalSkillsPlanner is original and sys.argv is argv
    assert not (tmp_path/'output').exists()


def test_existing_directory_unchanged_even_when_entry_rejected(tmp_path,monkeypatch):
    p=tmp_path/'old';p.mkdir();(p/'marker.txt').write_text('old')
    monkeypatch.setattr(sys,'argv',['run','--output-dir',str(p),'--prepare-only'])
    with pytest.raises(FileExistsError,match='preserve_prior'):q.main()
    assert [f.name for f in p.iterdir()]==['marker.txt'] and (p/'marker.txt').read_text()=='old'


def test_cannot_turn_on_quantity_supplement_when_skills_off(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['run','--quantity-evidence','on','--skills','off','--incremental-skills','on','--output-dir',str(tmp_path/'new'),'--prepare-only'])
    with pytest.raises(ValueError,match='requires_incremental_skills_on'):q.main()
    assert not (tmp_path/'new').exists()
