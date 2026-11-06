import json
from audit_chat_accounting import audit

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def fixture(tmp_path):
 save(tmp_path/'protocol.json',dict(arms=['a'],planned_scenes=[1],planned_attempts=1,models=['kimi-k3.1-pro'],transport='chat_completions'))
 save(tmp_path/'launcher_result.json',dict(status='completed'))
 save(tmp_path/'a/realtime_entry.json',dict(enabled=False))
 save(tmp_path/'a/movies/s/attempt_result.json',dict(scenario_index=1,executed_steps=1,collision=False,status='completed'))
 (tmp_path/'a/movies/s/execution_trace.jsonl').write_text('\n'.join(json.dumps(dict(kind='state',step=i)) for i in [0,1]))
 call=dict(model_requested='kimi-k3.1-pro',scenario_index=1,messages=[dict(role='user',content='plan')],usage=dict(total_tokens=17))
 (tmp_path/'a_api_calls.jsonl').write_text(json.dumps(call)+'\n')
 return call

def test_chat_usage_and_no_realtime_required(tmp_path):
 fixture(tmp_path);r=audit(tmp_path)
 assert r['accounting_verified'] and r['groups']['a']['observed_tokens']==17
 assert r['groups']['a']['realtime_connections']==0

def test_wrong_enabled_entry_rejected(tmp_path):
 fixture(tmp_path);save(tmp_path/'a/realtime_entry.json',dict(enabled=True))
 assert not audit(tmp_path)['accounting_verified']

def test_budget_not_relaxed(tmp_path):
 c=fixture(tmp_path);(tmp_path/'a_api_calls.jsonl').write_text((json.dumps(c)+'\n')*5)
 assert not audit(tmp_path)['accounting_verified']

def test_unexpected_realtime_rejected(tmp_path):
 c=fixture(tmp_path);(tmp_path/'a/realtime_calls.jsonl').write_text(json.dumps(dict(messages=c['messages'],model_requested='kimi-k3.1-pro',events=[],response_create_count=0))+'\n')
 assert not audit(tmp_path)['accounting_verified']
