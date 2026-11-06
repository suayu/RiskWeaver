"""模拟WebSocket验证预算、身份、用量及凭据隔离，不调用真实服务。"""
import json
import pytest
from transport import RealtimeClient, MODEL, RealtimeServiceError


class Socket:
    def __init__(self,model=MODEL,usage=True,status='completed'):
        self.sent=[]
        done={'id':'r1','status':status,'output':[{'content':[{'type':'text','text':'{"attack":false}'}]}]}
        if usage:done['usage']={'input_tokens':10,'output_tokens':5,'total_tokens':15}
        self.events=iter([{'type':'session.created','session':{'model':model}},
            {'type':'session.updated'},{'type':'conversation.item.created'},
            {'type':'response.text.delta','delta':'do not duplicate'},
            {'type':'response.done','response':done}])
        self.closed=False
    def settimeout(self,n):assert n>0
    def send(self,data):self.sent.append(json.loads(data))
    def recv(self):return json.dumps(next(self.events))
    def close(self):self.closed=True


def call(client):
    return client.chat.completions.create(model=MODEL,messages=[{'role':'user','content':'plan'}],
                                         response_format={'type':'json_object'})


def test_one_response_and_complete_usage(tmp_path):
    sock=Socket();client=RealtimeClient('secret-test',tmp_path,connector=lambda *a,**k:sock)
    r=call(client)
    assert r.usage.model_dump()['total_tokens']==15
    assert r.choices[0].message.content=='{"attack":false}'
    assert sum(e['type']=='response.create' for e in sock.sent)==1
    assert sock.closed
    assert 'secret-test' not in (tmp_path/'realtime_calls.jsonl').read_text()


@pytest.mark.parametrize('options,error',[({'model':'other'},ValueError),({'usage':False},ValueError),
                                        ({'status':'incomplete'},RealtimeServiceError)])
def test_no_silent_success_or_retry(tmp_path,options,error):
    sock=Socket(**options);connections=[]
    def connect(*a,**k):connections.append(1);return sock
    with pytest.raises(error):call(RealtimeClient('secret-test',tmp_path,connector=connect))
    assert len(connections)==1 and sock.closed


def test_auto_retry_refused(tmp_path):
    client=RealtimeClient('secret-test',tmp_path)
    with pytest.raises(ValueError):client.with_options(max_retries=1)


def test_wrong_model_and_multimodal_rejected_before_connect(tmp_path):
    def forbidden(*a,**k):raise AssertionError('network')
    client=RealtimeClient('secret-test',tmp_path,connector=forbidden)
    with pytest.raises(ValueError):client.create(model='other',messages=[])
    with pytest.raises(ValueError):client.create(model=MODEL,messages=[{'role':'user','content':[]}])
