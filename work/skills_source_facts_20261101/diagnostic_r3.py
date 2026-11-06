"""强制思考模型的有界参数诊断；R1/R2保留，不能归因于技能文字单独收益。"""
import argparse
import hashlib
import json
import time
import diagnostic as base

base.PROTOCOL = base.ROOT/'work/skills_source_facts_protocol_20261101_r4'
base.OUTPUT = base.ROOT/'experiments/skills_source_facts_frozen4_diagnostic_20261101_r3'
OriginalPlanner = base.base.LLMAdversarialPlanner


class BoundedThinkingPlanner(OriginalPlanner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.client is None:return
        self.client=self.client.with_options(timeout=180,max_retries=0)
        create=self.client.chat.completions.create
        def bounded(*args,**kwargs):
            values=dict(kwargs,max_tokens=4096)
            row=dict(model=values.get('model'),max_tokens=4096,timeout_seconds=180,
                SDK_retries=0,extra_body=values.get('extra_body'),started_at=time.time(),
                messages_sha256=hashlib.sha256(json.dumps(values.get('messages'),ensure_ascii=False,
                    sort_keys=True,allow_nan=False).encode()).hexdigest())
            with (base.OUTPUT/'transport_parameters.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(row,ensure_ascii=False)+'\n')
            return create(*args,**values)
        self.client.chat.completions.create=bounded


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:
        base.prepare();path=base.PROTOCOL/'protocol.json';protocol=json.loads(path.read_text())
        order=['glm-5.3','qwen3.8-2.4t-a95b','deepseek-v4.1-flash-1017']
        protocol['frozen_model_pool']=[m for m in order if m in protocol['model_source']['active_models']]
        protocol['common_transport_parameters']=dict(timeout_seconds=180,max_tokens=4096,SDK_retries=0)
        protocol['scope']='known_counterexamples_and_transport_parameter_diagnostic; not_skill_only_gain_or_generalization'
        protocol['preceding_runs_preserved']=['skills_source_facts_frozen4_diagnostic_20261101_r1','skills_source_facts_frozen4_diagnostic_20261101_r2']
        base.base.save(path,protocol)
        print(json.dumps(dict(status='prepared',models=protocol['frozen_model_pool'],transport=protocol['common_transport_parameters'],API_requests=0)))
    else:
        base.base.LLMAdversarialPlanner=BoundedThinkingPlanner
        base.run()
