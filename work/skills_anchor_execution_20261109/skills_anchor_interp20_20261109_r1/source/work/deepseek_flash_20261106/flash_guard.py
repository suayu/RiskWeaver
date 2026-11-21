"""固定DeepSeek官方Flash，拒绝Pro及任何跨模型回退。"""
import json,time
from urllib.parse import urlsplit
MODEL='deepseek-flash'
def parameters(kwargs):
    if kwargs.get('model')!=MODEL:raise ValueError('only_official_deepseek_flash_allowed')
    out=dict(kwargs);out['max_tokens']=8192;out['response_format']={'type':'json_object'}
    out['extra_body']={'thinking':{'type':'disabled'}}
    return out

def check_returned(model):
    if model!=MODEL:raise ValueError('returned_model_is_not_verified_flash')

def install(output):
    import openai
    original=openai.OpenAI.__init__
    def init(client,*args,**kwargs):
        url=urlsplit(str(kwargs.get('base_url','')))
        if url.scheme!='https' or url.hostname!='api.deepseek.com':raise ValueError('official_deepseek_endpoint_required')
        kwargs['max_retries']=0;original(client,*args,**kwargs)
        create=client.chat.completions.create
        def guarded(*a,**kw):
            response=create(*a,**parameters(kw))
            record=dict(requested=kw['model'],returned=response.model,endpoint='https://api.deepseek.com',paid_authorized=True,pro_fallback=False,time=time.time())
            with (output.parent/(output.name+'_flash_model_checks.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
            check_returned(response.model);return response
        client.chat.completions.create=guarded
    openai.OpenAI.__init__=init
    return lambda:setattr(openai.OpenAI,'__init__',original)
