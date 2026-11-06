"""独立文本Realtime适配器，维持一次create对应一次模型响应，不自动重试。"""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import uuid
import websocket
from openai.types.chat import ChatCompletion

HOST = 'ws-qvq9xoxtkn7fpem7.cn-beijing.maas.aliyuncs.com'
MODEL = 'qwen3.8-omni-flash-realtime'


class RealtimeServiceError(RuntimeError):
    def __init__(self, body):
        self.body = body
        super().__init__(json.dumps(body, ensure_ascii=False))


class RealtimeClient:
    def __init__(self, api_key, directory, timeout=180, connector=None):
        self._key = api_key
        self.directory = Path(directory)
        self.timeout = timeout
        self.connector = connector or websocket.create_connection
        self.base_url = f'wss://{HOST}/api-ws/v1/realtime'
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, timeout=180, max_retries=0):
        if max_retries != 0:
            raise ValueError('automatic_retries_prohibited')
        return RealtimeClient(self._key, self.directory, timeout, self.connector)

    def create(self, *, model, messages, max_tokens=8192, **kwargs):
        if model != MODEL:
            raise ValueError('unfrozen_realtime_model')
        if set(kwargs)-{'response_format','extra_body'}:
            raise ValueError('unsupported_realtime_parameters')
        if not messages or any(m.get('role') not in ('system','user','assistant') or
                               not isinstance(m.get('content'),str) for m in messages):
            raise ValueError('text_messages_required')
        self.directory.mkdir(parents=True,exist_ok=True)
        request_id = uuid.uuid4().hex
        events = []
        started = time.monotonic()
        sock = None
        record = dict(request_id=request_id, model_requested=model, response_create_count=0,
                      messages=messages, max_tokens=max_tokens, timeout_seconds=self.timeout,
                      JSON_mode='prompt_only_local_parser_validates', SDK_retries=0)

        def receive():
            remaining = self.timeout-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('realtime_request_deadline')
            sock.settimeout(remaining)
            event = json.loads(sock.recv())
            events.append(event)
            if event.get('type')=='error':
                raise RealtimeServiceError(event.get('error',{}))
            return event

        def send(event):
            events.append(dict(direction='client',**event))
            sock.send(json.dumps(event,ensure_ascii=False))

        try:
            sock = self.connector(f'{self.base_url}?model={MODEL}',
                                  header=['Authorization: Bearer '+self._key],
                                  timeout=min(15,self.timeout))
            while True:
                event = receive()
                if event.get('type')=='session.created':
                    if event.get('session',{}).get('model')!=MODEL:
                        raise ValueError('returned_model_mismatch')
                    break
            # Realtime没有沿用Chat JSON模式，明确采用提示与原解析器验证。
            send({'type':'session.update','session':{'modalities':['text'],
                  'turn_detection':None,'enable_search':False,'max_tokens':max_tokens,
                  'instructions':'Return one JSON object only. Follow the supplied planning contract. No markdown.'}})
            while receive().get('type')!='session.updated':
                pass
            for message in messages:
                send({'type':'conversation.item.create','item':{'type':'message',
                      'role':message['role'],'content':[{'type':'input_text','text':message['content']}]}})
                while receive().get('type')!='conversation.item.created':
                    pass
            record['response_create_count'] = 1
            send({'type':'response.create'})
            while True:
                event = receive()
                if event.get('type')=='response.done':
                    response = event['response']
                    break
            if response.get('status')!='completed':
                raise RealtimeServiceError({'type':'incomplete_response','status':response.get('status')})
            # 使用最终输出，避免把delta与done的相同文字重复拼接。
            text=''.join(c.get('text','') for item in response.get('output',[])
                         for c in item.get('content',[]) if c.get('type')=='text')
            usage=response.get('usage')
            if not text or not isinstance(usage,dict):
                raise ValueError('missing_text_or_usage')
            record.update(model_returned=MODEL,raw_usage=usage,content=text)
            return ChatCompletion(id=response['id'],object='chat.completion',created=int(time.time()),
                model=MODEL, choices=[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':text}}],
                usage={'prompt_tokens':usage['input_tokens'],'completion_tokens':usage['output_tokens'],
                       'total_tokens':usage['total_tokens']})
        except Exception as error:
            record['error_type']=type(error).__name__
            raise
        finally:
            if sock:
                sock.close()
            record['elapsed_seconds']=time.monotonic()-started
            record['events']=events
            # 不写认证头，完整事件中也替换可能回显的密钥。
            data=json.dumps(record,ensure_ascii=False).replace(self._key,'[REDACTED]')
            with (self.directory/'realtime_calls.jsonl').open('a',encoding='utf-8') as file:
                file.write(data+'\n')
