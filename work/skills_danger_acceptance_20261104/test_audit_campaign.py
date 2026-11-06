"""验收入口反例：输入、源码和视频缺失不能被汇总成成功。"""
import json
import sys
from types import SimpleNamespace
import audit_campaign as module
from test_danger_acceptance import fixture


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def prepare(tmp_path,monkeypatch):
    root=tmp_path/'experiment';root.mkdir()
    scene=tmp_path/'input.pkl';scene.write_bytes(b'frozen scene')
    source=root/'source_snapshot/code.py';source.parent.mkdir();source.write_text('# frozen')
    manifest=dict(original_scenario_indices=[1],scenario_files=[str(scene)],scenario_sha256=[module.digest(scene)])
    save(root/'manifest.json',manifest)
    save(root/'protocol.json',dict(arms=['a'],planned_scenes=[1],source_count=1,
                                manifest_sha256=module.digest(root/'manifest.json')))
    save(root/'source_sha256.json',{'code.py':module.digest(source)})
    save(root/'launcher_result.json',dict(status='completed'))
    save(root/'runtime_source_integrity.json',dict(verified=True))
    save(root/'a/event_joint_entry.json',dict(actual_config_verified=True))
    attempt,trace=fixture(True);directory=root/'a/movies/scenario_001'
    save(directory/'attempt_result.json',attempt)
    (directory/'execution_trace.jsonl').write_text('\n'.join(json.dumps(r) for r in trace),encoding='utf-8')
    (directory/'movie.mp4').write_bytes(b'test double')
    monkeypatch.setitem(sys.modules,'audit_realtime_pair',SimpleNamespace(audit=lambda root:dict(accounting_verified=True)))
    monkeypatch.setattr(module,'video_check',lambda p,expected_states=None:dict(path=str(p),verified=True))
    return root,scene


def test_clean_development_audit_never_completes20(tmp_path,monkeypatch):
    root,_=prepare(tmp_path,monkeypatch)
    report=module.audit(root,tmp_path)
    summary=report['groups']['a']['summary']
    assert summary['effective_danger_scenes']==1
    assert summary['campaign_integrity_verified']
    assert not summary['target_18_of20_met']


def test_changed_input_invalidates_campaign(tmp_path,monkeypatch):
    root,scene=prepare(tmp_path,monkeypatch);scene.write_bytes(b'changed')
    report=module.audit(root,tmp_path)
    assert not report['groups']['a']['summary']['campaign_integrity_verified']
    assert any(s.startswith('input_scene_sha_mismatch') for s in report['audit_issues'])


def test_missing_source_snapshot_invalidates_campaign(tmp_path,monkeypatch):
    root,_=prepare(tmp_path,monkeypatch);(root/'source_snapshot/code.py').unlink()
    report=module.audit(root,tmp_path)
    assert not report['groups']['a']['summary']['campaign_integrity_verified']


def test_missing_video_does_not_certify_episode(tmp_path,monkeypatch):
    root,_=prepare(tmp_path,monkeypatch)
    next(root.rglob('*.mp4')).unlink()
    report=module.audit(root,tmp_path)
    assert report['groups']['a']['summary']['effective_danger_scenes']==0
    assert 'video_verification_failed' in report['groups']['a']['audit_issues']
