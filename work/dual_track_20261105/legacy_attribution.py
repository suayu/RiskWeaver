"""旧自由规划无request_id时的只读派生关联；原始trace不改写。"""
import copy,json


def recover(rows,calls,scene):
    rows=copy.deepcopy(rows);plans={};bindings=[];unresolved=[];active=None
    for row in rows:
        if row.get('kind')=='llm_output':
            p=row.get('validated_plan') or {}
            if p.get('request_id') or not (p.get('attack') or p.get('obstacle_plan')):continue
            matches=[]
            for c in calls:
                if c.get('scenario_index')!=scene or c.get('source_step')!=row['step'] or not c.get('model_returned'):continue
                try:raw=json.loads(c.get('content',''))
                except (ValueError,TypeError):continue
                if all(raw.get(k)==p.get(k) for k in ['attack','attack_target_id','strategy','reason']) and len(raw.get('obstacle_plan',[]))==len(p.get('obstacle_plan',[])):
                    matches.append(c)
            if len(matches)==1:
                ident=matches[0]['request_id'];p['request_id']=ident;plans[row['step']]=p
                bindings.append(dict(step=row['step'],request_id=ident,basis='unique_scene_source_step_and_raw_plan_identity_reason; coordinates_may_be_transformed'))
            else:unresolved.append(dict(step=row['step'],reason='ambiguous_or_unmatched_sdk',matches=len(matches)))
        elif row.get('kind')=='plan_accepted':
            active=None;p=plans.get(row['step'])
            if p and p.get('attack') and p.get('attack_target_id')==row.get('target_id'):
                active=(row['step'],p);row['request_id']=p['request_id']
        elif row.get('kind')=='attack_execution' and row.get('executed') and not row.get('request_id'):
            if active:
                start,p=active
                if p['attack_target_id']==row.get('target_id') and start<row['step']<=start+10 and row.get('diffusion_source_step',-1)>=start:
                    row['request_id']=p['request_id']
        elif row.get('kind')=='static_attack_execution' and not row.get('request_id'):
            p=plans.get(row['step'])
            if p and p.get('obstacle_plan'):row['request_id']=p['request_id']
    return rows,dict(bindings=bindings,unresolved=unresolved,original_trace_modified=False,
        scope='conservative_derived_journal_association; not_new_execution_or_semantic_certificate')
