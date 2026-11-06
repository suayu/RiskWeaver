"""防止用SHA字符串、错误内容或无归属请求代替真实文档全文。"""
import hashlib
import json
from audit_skill_payloads import check_document,payload_contains


def test_exact_JSON_escaped_full_document_matches(tmp_path):
    text='# 技能\n正确引用参与者。\n';path=tmp_path/'SKILL.md';path.write_text(text,encoding='utf-8')
    call=dict(request_id='physical1',messages=[dict(content=json.dumps(dict(rule=text),ensure_ascii=False))])
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    assert check_document(path,expected,[call])['request_ids']==['physical1']


def test_hash_alone_is_not_full_rule(tmp_path):
    path=tmp_path/'SKILL.md';path.write_text('完整规则正文',encoding='utf-8')
    expected=hashlib.sha256(path.read_bytes()).hexdigest()
    result=check_document(path,expected,[dict(request_id='p',messages=[dict(content=expected)])])
    assert not result['verified'] and result['issue']=='full_text_missing_from_actual_SDK'


def test_changed_rule_rejected_even_if_sent(tmp_path):
    path=tmp_path/'SKILL.md';path.write_text('已更改',encoding='utf-8')
    result=check_document(path,'0'*64,[dict(request_id='p',messages=[dict(content='已更改')])])
    assert not result['verified'] and result['issue']=='document_sha_mismatch'


def test_missing_or_unassociated_request_not_certified(tmp_path):
    path=tmp_path/'SKILL.md'
    assert check_document(path,'0'*64,[])['issue']=='document_missing'
    path.write_text('规则',encoding='utf-8')
    assert not check_document(path,hashlib.sha256(path.read_bytes()).hexdigest(),[])['verified']
