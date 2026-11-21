"""使用原四组入口和原排队循环核验分母、隔离与前置阻断；不运行模型。"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import pytest
import launch as q


def test_actual_four_arm_dispatch_records_40_and_isolates_failure(tmp_path,monkeypatch):
    previous=q.previous;launcher=previous.launcher
    (tmp_path/'work').mkdir()
    monkeypatch.setattr(launcher,'ROOT',tmp_path)
    protocol=tmp_path/'protocol';protocol.mkdir();(protocol/'source').mkdir()
    arms={n:{} for n in ['legacy_shared_recovery','catalog_no_knowledge','skills_no_expansion','skills_full_shared_recovery']}
    (protocol/'protocol.json').write_text(json.dumps(dict(required_smoke_receipts=[],model_source={},frozen_model_pool=['kimi-k3'])))
    (protocol/'source_sha256.json').write_text('{}')
    (protocol/'development10.json').write_text(json.dumps(dict(scenario_count=10,original_scenario_indices=list(range(10)))))
    (protocol/'prepared_arms.json').write_text(json.dumps(arms))
    out=tmp_path/'out';seen=[]
    monkeypatch.setattr(launcher,'verify_sources',lambda *args:dict(verified=True))
    monkeypatch.setattr(launcher,'idle_gpus',lambda reserved:[g for g in ['1','2'] if g not in reserved])
    monkeypatch.setattr(launcher,'child_command',lambda prepared,manifest,output:['fake',output.name])
    monkeypatch.setattr(previous,'audit_four_arm',lambda root:dict(development_gate=dict(passed=False)))
    def popen(command,**kwargs):
        seen.append(command[-1])
        return SimpleNamespace(pid=100+len(seen),wait=lambda:1 if command[-1]=='catalog_no_knowledge' else 0)
    monkeypatch.setattr(launcher.subprocess,'Popen',popen)
    monkeypatch.setattr(sys,'argv',['launch','--protocol',str(protocol),'--stage','development','--output-dir',str(out),'--free-protection-confirmed'])
    oldsave=launcher.save
    q.main()
    assert launcher.save is oldsave and set(seen)==set(arms) and len(seen)==4
    record=json.loads((out/'launch.json').read_text())
    assert record['planned_attempts']==40 and record['planned_scenes_per_arm']==10
    final=json.loads((out/'launcher_result.json').read_text())
    assert final['status']=='completed_with_group_failures' and len(final['groups'])==4
    assert final['acceptance_passed'] is False


def test_real_prerequisite_guard_still_blocks_before_dispatch(tmp_path,monkeypatch):
    p=tmp_path/'protocol';p.mkdir()
    (p/'protocol.json').write_text(json.dumps(dict(required_smoke_receipts=[dict(experiment='missing',arm='incremental_full',scenario_index=74)])))
    monkeypatch.setattr(q.previous,'smoke_prerequisite',lambda *args:dict(passed=False))
    monkeypatch.setattr(sys,'argv',['launch','--protocol',str(p),'--stage','development'])
    oldsave=q.previous.launcher.save
    with pytest.raises(ValueError,match='smoke_prerequisite_not_passed'):q.main()
    assert q.previous.launcher.save is oldsave
    assert not (tmp_path/'out').exists()


def test_duplicate_manifest_is_rejected_before_accepting_denominator(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(scenario_count=2,original_scenario_indices=[74,74])))
    calls=[];launcher=SimpleNamespace(save=lambda *args:calls.append(args))
    restore=q.install_attempt_count(launcher)
    try:
        with pytest.raises(ValueError,match='duplicate'):launcher.save(tmp_path/'launch.json',dict(arms=['a','b','c','d']))
        assert not calls
    finally:restore()
