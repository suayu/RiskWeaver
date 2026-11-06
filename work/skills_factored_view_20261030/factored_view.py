"""可逆的请求视图因式分解；只合并精确相同字段，不改身份、数值或候选。"""
from pathlib import Path
import copy,hashlib,json

HERE=Path(__file__).resolve().parent
MARKER='_factored_defaults_v1'


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def common_fields(rows):
    """类型和值均按严格JSON相等；True、1、1.0不得因Python相等被合并。"""
    if len(rows)<2:return {}
    common={}
    for key in sorted(set.intersection(*(set(row) for row in rows))):
        values=[row[key] for row in rows]
        if all(canonical(v)==canonical(values[0]) for v in values[1:]):common[key]=copy.deepcopy(values[0])
        elif all(isinstance(v,dict) for v in values):
            nested=common_fields(values)
            if nested:common[key]=nested
    return common


def subtract(row,defaults):
    """留空字典标记继承，不把缺失/空值转成零。"""
    for key,value in defaults.items():
        if canonical(row[key])==canonical(value):del row[key]
        elif isinstance(value,dict) and isinstance(row[key],dict):subtract(row[key],value)
        else:raise ValueError('nonidentical_field_cannot_be_factored')


def restore(row,defaults):
    for key,value in defaults.items():
        if key not in row:row[key]=copy.deepcopy(value)
        elif isinstance(value,dict) and isinstance(row[key],dict):restore(row[key],value)
        else:raise ValueError('conflicting_factored_default')


def rows_for(context,name):
    """分组路径固定，不由模型提供路径或执行任意代码。"""
    if name=='candidates':return context.get('candidates',[])
    if name=='priority_outer':return list(context.get('priority_evidence',{}).values())
    if name.startswith('priority_fact:'):
        kind=name.split(':',1)[1]
        return [row['facts'][cid+':'+kind] for cid,row in context.get('priority_evidence',{}).items() if cid+':'+kind in row.get('facts',{})]
    if name=='rolling_candidate_facts':return list(context.get('rolling_intent',{}).get('candidate_facts',{}).values())
    if name=='comparison_rows':return list(context.get('comparison_facts',{}).get('per_candidate',{}).values())
    if name=='fractional_rows':return list(context.get('temporal_candidate_evidence',{}).get('per_candidate_fractional_minimum',{}).values())
    raise ValueError('unknown_factored_group')


GROUPS=['candidates',*[f'priority_fact:{kind}' for kind in ['actor','validity','active','safety','after','contrast']],
        'priority_outer','rolling_candidate_facts','comparison_rows','fractional_rows']


def encode_payload(payload):
    original=copy.deepcopy(payload);output=copy.deepcopy(payload);context=output['context']
    if MARKER in context:raise ValueError('already_factored_or_reserved_marker')
    metadata=[]
    for group in GROUPS:
        rows=rows_for(context,group)
        if not all(isinstance(row,dict) for row in rows):raise ValueError('unexpected_record_type')
        defaults=common_fields(rows)
        if group=='priority_outer':defaults={k:v for k,v in defaults.items() if k!='facts'}
        if defaults:
            for row in rows:subtract(row,defaults)
            metadata.append(dict(group=group,defaults=defaults,record_count=len(rows)))
    context[MARKER]=dict(version='lossless_json_defaults_v1',groups=metadata,
        values_not_rounded_or_imputed=True,candidate_order_not_changed=True,response_schema_unchanged=True)
    if canonical(expand_payload(output))!=canonical(original):raise ValueError('lossless_request_view_roundtrip_failed')
    return output


def expand_payload(payload):
    output=copy.deepcopy(payload);metadata=output['context'].pop(MARKER)
    if metadata['version']!='lossless_json_defaults_v1':raise ValueError('unknown_factored_version')
    for item in reversed(metadata['groups']):
        rows=rows_for(output['context'],item['group'])
        if len(rows)!=item['record_count']:raise ValueError('factored_catalog_cardinality_changed')
        for row in rows:restore(row,item['defaults'])
    return output


def render(prompt):
    """作用于既有传输压缩后的真实请求；全内部上下文和SDK请求应分开记录。"""
    try:prefix,raw=prompt.rsplit('\n',1);payload=json.loads(raw)
    except (ValueError,TypeError):return prompt,dict(supported=False,reason='not_structured_skills_prompt')
    context=payload.get('context',{})
    if not isinstance(context.get('priority_evidence'),dict):return prompt,dict(supported=False,reason='not_priority_grounding_interface')
    factored=encode_payload(payload)
    skill=(HERE/'SKILL.md').read_text(encoding='utf-8').split('---',2)[-1].strip()
    result=prefix+'\n'+skill+'\n'+json.dumps(factored,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    return result,dict(supported=True,version='lossless_json_defaults_v1',original_wire_bytes=len(prompt.encode('utf-8')),
        factored_wire_bytes=len(result.encode('utf-8')),original_wire_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        factored_wire_sha256=hashlib.sha256(result.encode('utf-8')).hexdigest(),payload_roundtrip_verified=True,
        identity_fields_unchanged=all(payload[k]==factored[k] for k in ['request_id','state_hash','catalog_hash']),
        semantic_model_performance_unverified=True,token_savings_not_inferred_from_bytes=True)


def install_factored_view(engine,enabled=False):
    """默认零改动；只包装真实请求边界，原回复/验证器/调用预算全部沿用。"""
    if not isinstance(enabled,bool):raise ValueError('explicit_boolean_view_switch_required')
    if not enabled:return engine
    plan=engine.plan
    def controlled(planner,*args,**kwargs):
        request=planner._request_attack_plan;records=[]
        def outgoing(prompt,scene_image=None):
            view,receipt=render(prompt);records.append(receipt)
            return request(view,scene_image=scene_image)
        planner._request_attack_plan=outgoing
        try:return plan(planner,*args,**kwargs)
        finally:
            planner._request_attack_plan=request
            planner.last_trace['factored_wire_view']=dict(enabled=True,records=records,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),raw_response_not_rewritten=True,
                no_extra_requests=True,not_installed_in_existing_frozen_experiments=True)
    engine.plan=controlled;return engine
