"""合成日志仅检验Chat接线归属，不作为模型性能证据。"""
from pathlib import Path
import json,hashlib
from audit_skill_payloads import audit,RULES

def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v),encoding='utf-8')

def prepare(tmp):
 root=tmp/'experiment';s=root/'source_snapshot/skills/riskweaver';files={}
 for name in RULES:
  p=s/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('Rule content '+name,encoding='utf-8');files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 save(s/'manifest.json',dict(files=files))
 save(root/'source_sha256.json',{'skills/riskweaver/manifest.json':hashlib.sha256((s/'manifest.json').read_bytes()).hexdigest()})
 save(root/'protocol.json',dict(arms=['a'],transport='chat_completions'))
 request='highlevel-request-123456';text=json.dumps(dict(request_id=request,rules=[(s/name).read_text(encoding='utf-8') for name in RULES]))
 save(root/'a_api_calls.jsonl',dict(request_id='physical-1',model_returned='kimi-k3.1-pro',messages=[dict(role='user',content=text)]))
 row=dict(kind='llm_output',step=0,planner_trace=dict(request_id=request,skill_manifest=dict(documents=[dict(path=RULES[0],sha256=files[RULES[0]])])))
 save(root/'a/movies/s/execution_trace.jsonl',row)
 return root,tmp/'supplemental'

def test_chat_actual_journal_used_without_realtime(tmp_path):
 root,supp=prepare(tmp_path);r=audit(root,supp)
 assert r['verified'] and r['groups']['a']['physical_requests']==1
 assert r['groups']['a']['events'][0]['associated_physical_ids']==['physical-1']

def test_unrelated_request_cannot_certify_rule_attribution(tmp_path):
 root,supp=prepare(tmp_path);p=root/'a_api_calls.jsonl';v=json.loads(p.read_text());v['messages'][0]['content']=v['messages'][0]['content'].replace('highlevel-request-123456','another-event');save(p,v)
 assert not audit(root,supp)['verified']

def test_missing_chat_journal_not_treated_as_complete(tmp_path):
 root,supp=prepare(tmp_path);(root/'a_api_calls.jsonl').unlink()
 assert not audit(root,supp)['verified']
