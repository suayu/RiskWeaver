"""百炼Realtime单响应探测；不写默认模型配置，不记录认证头。"""
from pathlib import Path
import json
import os
import time
import websocket

HOST = 'ws-qvq9xoxtkn7fpem7.cn-beijing.maas.aliyuncs.com'
MODEL = 'qwen3.8-omni-flash-realtime'
OUT = Path('work/realtime_workspace_probe_20261104_r2')


def main():
    OUT.mkdir(exist_ok=False)
    key = os.environ['DASHSCOPE_API_KEY']
    started = time.monotonic()
    sock = None
    rows = []
    result = dict(model=MODEL, host=HOST, response_requests=0, success=False,
                  max_tokens=64, timeout_seconds=30, retries=0)

    def receive():
        # 所有接收共用总时限，防止持续事件使单次探测无限延长。
        remaining = 30 - (time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError('probe_deadline')
        sock.settimeout(remaining)
        event = json.loads(sock.recv())
        rows.append(event)
        if event.get('type') == 'error':
            raise ValueError('server_error')
        return event

    try:
        sock = websocket.create_connection(
            f'wss://{HOST}/api-ws/v1/realtime?model={MODEL}',
            header=['Authorization: Bearer '+key], timeout=15)
        while receive().get('type') != 'session.created':
            pass
        config = {'type':'session.update', 'session':{
            'modalities':['text'], 'turn_detection':None, 'enable_search':False,
            'instructions':'For this connection test, reply exactly {"probe":true} as JSON. No explanation.',
            'max_tokens':64}}
        rows.append({'direction':'client', **config})
        sock.send(json.dumps(config))
        while receive().get('type') != 'session.updated':
            pass
        # 服务端要求显式输入；单独创建用户文本项，不把系统配置当输入。
        item = {'type':'conversation.item.create','item':{'type':'message','role':'user',
                'content':[{'type':'input_text','text':'Return exactly {"probe":true}'}]}}
        rows.append({'direction':'client', **item})
        sock.send(json.dumps(item))
        while receive().get('type') != 'conversation.item.created':
            pass
        result['response_requests'] = 1
        sock.send(json.dumps({'type':'response.create'}))
        while True:
            event = receive()
            if event.get('type') == 'response.done':
                result['response'] = event.get('response')
                break
        text = ''.join(e.get('delta','') for e in rows if e.get('type')=='response.text.delta')
        result['text'] = text
        try:
            result['success'] = json.loads(text)=={'probe':True}
        except ValueError:
            pass
    except Exception as error:
        result['error_type'] = type(error).__name__
        result['http_status'] = getattr(error,'status_code',None)
        # 不打印异常对象或请求头；响应正文只保留服务端错误并替换凭据。
        body = getattr(error,'resp_body',None)
        if body:
            result['server_body'] = str(body).replace(key,'[REDACTED]')[:2000]
    finally:
        if sock:
            sock.close()
        result['elapsed_seconds'] = time.monotonic()-started
        raw = json.dumps(rows,ensure_ascii=False,indent=2).replace(key,'[REDACTED]')
        (OUT/'events.json').write_text(raw,encoding='utf-8')
        (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
