"""实际留出阶段循环回归：二十个固定状态臂、失败分母、错清单拒绝；零API。"""
from pathlib import Path
import json,sys
import pytest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import run_holdout as entry

PROTOCOL=entry.ROOT/'work/skills_candidate_local_holdout_protocol_20261030_r1'


def test_actual_twenty_state_loop_and_alternating_order_without_API(tmp_path,monkeypatch):
    calls=[]
    def trial(sample,mode,pool,cfg,out):
        calls.append((sample['scene'],mode));out.mkdir()
        return dict(scene=sample['scene'],mode=mode,accepted_plan=False,planner_service_failure=False,budget=dict(transport_calls=0))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(entry.base,'run_trial',trial)
    monkeypatch.setattr(sys,'argv',['entry','--protocol',str(PROTOCOL),'--output',str(tmp_path/'run'),'--free-protection-confirmed'])
    entry.main();result=json.loads((tmp_path/'run/results.json').read_text())
    expected=[(scene,mode) for i,scene in enumerate(entry.EXPECTED) for mode in (entry.ARMS if i%2==0 else list(reversed(entry.ARMS)))]
    assert calls==expected and len(set(calls))==20 and result['planned_state_arms']==20
    assert result['new_execution_frames']==0 and result['D'] is None and result['final_source_check']['verified']


def test_final_service_failure_keeps_nineteen_unattempted_in_twenty(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(entry.base,'run_trial',lambda sample,mode,*args:dict(scene=sample['scene'],mode=mode,planner_service_failure=True))
    monkeypatch.setattr(sys,'argv',['entry','--protocol',str(PROTOCOL),'--output',str(tmp_path/'failed'),'--free-protection-confirmed'])
    with pytest.raises(RuntimeError,match='final_planning'):entry.main()
    result=json.loads((tmp_path/'failed/results.json').read_text())
    assert result['planned_state_arms']==20 and result['unattempted_state_arms']==19 and len(result['results'])==1


def test_development_scenes_cannot_be_presented_as_holdout(tmp_path,monkeypatch):
    protocol=tmp_path/'bad_protocol';protocol.mkdir()
    value=json.loads((PROTOCOL/'protocol.json').read_text());value['scenes']=[53]*10
    (protocol/'protocol.json').write_text(json.dumps(value));(protocol/'source_sha256.json').write_text('{}')
    (protocol/'stage_result.json').write_text(json.dumps(dict(status='completed')))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(sys,'argv',['entry','--protocol',str(protocol),'--output',str(tmp_path/'never_created'),'--free-protection-confirmed'])
    with pytest.raises(ValueError,match='readiness'):entry.main()
    assert not (tmp_path/'never_created').exists()
