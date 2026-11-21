import os,json
from openai import OpenAI
c=OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'],base_url='https://api.deepseek.com',max_retries=0,timeout=30)
v=c.models.list().model_dump()
print(json.dumps({'base_url':'https://api.deepseek.com','models':[{k:r.get(k) for k in ['id','name','owned_by']} for r in v['data']]}))
