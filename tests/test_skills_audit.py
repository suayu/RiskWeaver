"""验收器必须保留失败分母、无效D，以及不同未来时间的锚点误差。"""
import json
from experiments.skills import audit


def write_case(root,scenario=0):
    directory=root/'arm'/'movies'/f'scenario_{scenario:03d}';directory.mkdir(parents=True)
    record=dict(scenario_index=scenario,executed_steps=30,status='completed',attack_plan_count=1,
                scenario_danger_score=None,collision=False,generation_failure=False)
    (directory/'attempt_result.json').write_text(json.dumps(record))
    (directory/'scene.mp4').write_bytes(b'probe_is_mocked_separately')
    rows=[dict(kind='state',step=i,ego=[i,10.,10.,0.],agents=[[i+.5,0.]],
               background_pairs=[],background_static_pairs=[],static_obstacles=[]) for i in range(31)]
    rows += [dict(kind='attack_execution',step=i,executed=True) for i in range(1,11)]
    rows += [dict(kind='prediction',step=0,attack_intent=dict(source_step=0,target_id=0,
        strategy='cut_in',duration=10,anchors=[[0.,0.],[10.,0.],[20.,0.],[30.,0.]])),
        dict(kind='execution_prediction',step=0,joint=dict(source_step=0,agent_ids=[0],
             positions_global=[[[i,0.] for i in range(1,33)]],metadata={'anchor_guidance_active':True})),
        dict(kind='llm_output',step=0,planner_wall_seconds=1.,planner_trace=dict(
             mode='skills_candidate_ranking',api_calls_before=0,api_calls_total=1,
             requests=[{'response':{}}],successful_model='test'))]
    (directory/'execution_trace.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    return directory/'attempt_result.json'


def test_forecast_anchors_are_time_aligned_and_expiry_disclosed(tmp_path,monkeypatch):
    path=write_case(tmp_path)
    monkeypatch.setattr(audit,'probe_video',lambda p:dict(valid=True,frame_count=31))
    result=audit.audit_attempt(path)
    assert result['actual_attack_frames']==10 and result['D'] is None
    errors=result['first_forecasts'][0]['anchor_errors']
    assert [e['predicted_error_m'] for e in errors]==[0.,0.,0.]
    assert [e['actual_error_m'] for e in errors]==[.5,.5,.5]
    assert [e['within_attack_validity'] for e in errors]==[True,False,False]
    assert result['completeness']['video_frames_match']


def test_missing_attempts_remain_in_denominator_and_do_not_pass(tmp_path,monkeypatch):
    write_case(tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps({'scenario_count':3}))
    (tmp_path/'launch.json').write_text(json.dumps({'arms':['arm']}))
    (tmp_path/'arm'/'exit_status.json').write_text(json.dumps({'exit_code':1}))
    monkeypatch.setattr(audit,'probe_video',lambda p:dict(valid=True,frame_count=31))
    result=audit.audit_experiment(tmp_path)['groups']['arm']
    assert result['actual_attack_coverage']==1/3 and result['unattempted_scenes']==2
    assert not result['execution_coverage_gate_pass'] and result['D_mean_valid_scenes'] is None


def test_corrupt_video_or_missing_state_is_not_complete(tmp_path,monkeypatch):
    path=write_case(tmp_path)
    trace=path.parent/'execution_trace.jsonl'
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    trace.write_text('\n'.join(json.dumps(r) for r in rows if not(r['kind']=='state' and r['step']==5)))
    monkeypatch.setattr(audit,'probe_video',lambda p:dict(valid=False,reason='corrupt'))
    result=audit.audit_attempt(path)
    assert result['completeness']['terminal_state_present']
    assert not result['completeness']['state_sequence_complete']
    assert not result['completeness']['video_frames_match']
