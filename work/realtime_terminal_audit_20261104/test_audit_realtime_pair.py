"""合成终态只检验审计逻辑，不作为真实LLM或仿真实验。"""
import json
from audit_realtime_pair import audit


def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d),encoding='utf-8')


def make_pair(tmp_path):
    save(tmp_path/'protocol.json',dict(arms=['old','skills'],planned_scenes=[74],planned_attempts=2,models=['m']))
    save(tmp_path/'launcher_result.json',dict(status='failed'))
    p=tmp_path/'old/movies/scenario_074'
    save(p/'attempt_result.json',dict(scenario_index=74,executed_steps=1,collision=False,status='completed',
                                     attack_executed_frames=0,scenario_danger_score=None))
    rows=[dict(kind='state',initial=True),dict(kind='state'),
          dict(kind='llm_output',validated_plan=dict(attack=True)),dict(kind='projection_rejected')]
    (p/'execution_trace.jsonl').write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf-8')
    return tmp_path


def test_highlevel_acceptance_is_separate_from_execution(tmp_path):
    r=audit(make_pair(tmp_path));e=r['groups']['old']['episodes'][0]
    assert e['highlevel_validated_attack_outputs']==1
    assert e['attack_execution_trace_frames']==0
    assert e['scenario_danger_score'] is None


def test_missing_arm_remains_in_denominator(tmp_path):
    r=audit(make_pair(tmp_path))
    assert r['planned_attempts']==2
    e=r['groups']['skills']['episodes'][0]
    assert e['attempted'] is False and e['collision'] is None


def test_control_events_are_not_extra_model_calls(tmp_path):
    make_pair(tmp_path)
    c=dict(messages=[dict(role='user',content='x')],model_requested='m',scenario_index=74)
    (tmp_path/'old_api_calls.jsonl').write_text(json.dumps(c)+'\n')
    t=dict(c,response_create_count=1,events=[dict(direction='client',type=x) for x in
        ['session.update','conversation.item.create','response.create']],raw_usage=dict(total_tokens=15),model_returned='m')
    path=tmp_path/'old/realtime_transport/realtime_calls.jsonl';path.parent.mkdir()
    path.write_text(json.dumps(t)+'\n')
    r=audit(tmp_path)
    assert r['groups']['old']['physical_calls']==1 and r['groups']['old']['response_creates']==1
    assert not r['audit_issues']


def test_missing_journal_is_detected(tmp_path):
    make_pair(tmp_path)
    (tmp_path/'old_api_calls.jsonl').write_text(json.dumps(dict(messages=[],model_requested='m'))+'\n')
    assert any('journals_disagree' in x for x in audit(tmp_path)['audit_issues'])
