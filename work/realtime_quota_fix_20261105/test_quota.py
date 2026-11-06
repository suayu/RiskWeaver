import json
import pytest
import websocket
from transport import RealtimeClient,RealtimeServiceError,MODEL
from test_transport import Socket,call
from policies.llm_adversarial_planner import LLMAdversarialPlanner

class ErrorSocket(Socket):
 def __init__(self,event):
  super().__init__();self.events=iter([{'type':'session.created','session':{'model':MODEL}},event])

@pytest.mark.parametrize('nested',[False,True])
def test_explicit_free_quota_classified_before_response(tmp_path,nested):
 body={'code':'AllocationQuota.FreeTierOnly','message':'The free quota has been exhausted'}
 sock=ErrorSocket({'type':'error','error':body} if nested else body)
 with pytest.raises(RealtimeServiceError) as e:call(RealtimeClient('test-only-key',tmp_path,connector=lambda *a,**k:sock))
 assert e.value.status_code==429
 assert LLMAdversarialPlanner._is_service_failure(e.value)
 assert LLMAdversarialPlanner._is_free_quota_exhausted(e.value)
 row=json.loads((tmp_path/'realtime_calls.jsonl').read_text())
 assert row['response_create_count']==0 and row['error_body']['code']=='AllocationQuota.FreeTierOnly'
 assert sock.closed and 'test-only-key' not in (tmp_path/'realtime_calls.jsonl').read_text()

def test_plain_disconnect_service_failure_without_permanent_disable(tmp_path):
 class Broken(Socket):
  def recv(self):raise websocket.WebSocketConnectionClosedException('connection lost')
 sock=Broken()
 with pytest.raises(RealtimeServiceError) as e:call(RealtimeClient('test-only-key',tmp_path,connector=lambda *a,**k:sock))
 assert LLMAdversarialPlanner._is_service_failure(e.value)
 assert not LLMAdversarialPlanner._is_free_quota_exhausted(e.value)
 row=json.loads((tmp_path/'realtime_calls.jsonl').read_text())
 assert row['error_body']['cause_type']=='WebSocketConnectionClosedException'

@pytest.mark.parametrize('code',['Throttling','AccessDenied'])
def test_other_provider_errors_do_not_disable_quota(tmp_path,code):
 sock=ErrorSocket({'code':code,'message':'service error'})
 with pytest.raises(RealtimeServiceError) as e:call(RealtimeClient('test-key',tmp_path,connector=lambda *a,**k:sock))
 assert LLMAdversarialPlanner._is_service_failure(e.value)
 assert not LLMAdversarialPlanner._is_free_quota_exhausted(e.value)


def test_common_pool_stops_after_single_explicit_quota(monkeypatch):
    p=object.__new__(LLMAdversarialPlanner)
    p.provider='dashscope';p.model_names=(MODEL,);p._preferred_model_index=0
    calls=[];disabled=[]
    def request(*a,**kw):
        calls.append(1)
        raise RealtimeServiceError({'code':'AllocationQuota.FreeTierOnly','message':'free quota exhausted'})
    def disable(name,*a,**kw):disabled.append(name);p.model_names=()
    p._request_attack_plan_once=request;p._disable_exhausted_model=disable
    monkeypatch.setattr(LLMAdversarialPlanner,'_exhausted_models_from_source',staticmethod(lambda source_path=None:set()))
    with pytest.raises(RealtimeServiceError):p._request_attack_plan('test')
    assert calls==[1] and disabled==[MODEL] and not p.model_names
    assert p.last_model_failures[0]['free_quota_exhausted']


@pytest.mark.parametrize('raw',['','{broken','[]','null'])
def test_empty_or_malformed_wire_is_service_failure(tmp_path,raw):
    class Raw(Socket):
        def recv(self):return raw
    sock=Raw()
    with pytest.raises(RealtimeServiceError) as e:call(RealtimeClient('dummy-key',tmp_path,connector=lambda *a,**k:sock))
    assert LLMAdversarialPlanner._is_service_failure(e.value)
    assert not LLMAdversarialPlanner._is_free_quota_exhausted(e.value)
    assert sock.closed


def test_credential_echo_redacted_in_exception_and_journal(tmp_path):
    sock=ErrorSocket({'code':'AccessDenied','message':'reflected-dummy-key'})
    with pytest.raises(RealtimeServiceError) as e:call(RealtimeClient('dummy-key',tmp_path,connector=lambda *a,**k:sock))
    assert 'dummy-key' not in str(e.value)
    assert 'dummy-key' not in (tmp_path/'realtime_calls.jsonl').read_text()
