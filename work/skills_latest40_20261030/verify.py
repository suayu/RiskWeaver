"""当前公共与可选时序验证器在原40输入的确定性回归；不创建LLM客户端或新仿真。"""
from pathlib import Path
from types import SimpleNamespace as NS
from collections import Counter
import argparse,copy,hashlib,json,os,sys
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_temporal_candidates_20261029')]
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.intent_validation import validate_world_intent,canonical_hash
from policies.skills_planner import SkillsAttackPlanner
from policies.rag_runtime import current_facts
from policies.rag_retrieval import CaseRetriever
from temporal_intents import guarded_fractional_validator


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform_state(state,angle,shift):
    """刚性坐标变换同时变换位置、速度、朝向、路线和历史；不改物理关系。"""
    moved=copy.deepcopy(state);c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s],[s,c]])
    def actor(values):
        row=list(values);row[:2]=(np.array(row[:2])@rot.T+shift).tolist();row[2:4]=(np.array(row[2:4])@rot.T).tolist();row[4]+=angle;return row
    moved['ego_state']=actor(moved['ego_state']);moved['route']=(np.array(moved['route'])@rot.T+shift).tolist()
    for item in moved['agents']:
        item['state']=actor(item['state'])
        for h in item.get('history',[]):
            if len(h.get('state',[]))>=5:h['state']=actor(h['state'])
    for obstacle in moved.get('static_obstacles',[]):
        obstacle['center']=(np.array(obstacle['center'])@rot.T+shift).tolist();obstacle['yaw']+=angle
    return moved,rot


def compare_facts(a,b):
    """不把null补零；只比较应在世界系变换下保持不变的当前事实。"""
    if a['scene_conditions']!=b['scene_conditions']:return False
    if a['candidate_strategies']!=b['candidate_strategies'] or a['candidate_families']!=b['candidate_families']:return False
    for x,y in zip(a['actors'],b['actors']):
        for k in ['current_target_id','longitudinal_relation','lateral_relation','motion_direction']:
            if x[k]!=y[k]:return False
        for k in ['current_distance_m','right_m','forward_m','radial_gap_rate_mps']:
            if x[k] is None or y[k] is None:
                if x[k] is not y[k]:return False
            elif abs(x[k]-y[k])>1e-8:return False
    for k,x in a['numeric'].items():
        y=b['numeric'][k]
        if x is None or y is None:
            if x is not y:return False
        elif abs(x-y)>1e-8:return False
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only_required')
    if args.output.exists():raise FileExistsError('preserve_verification_output')
    source=ROOT/'work/skills_stage_a_verification_20261023/original40_inputs.json'
    old=ROOT/'experiments/skills_stage_a_replay_20261020_r1/records.json'
    inputs=json.loads(source.read_text());historical={(r['scene'],r['step']):r for r in json.loads(old.read_text())}
    assert len(inputs['records'])==40 and len(historical)==40 and inputs['source_unchanged']
    watched=[source,old,Path(__file__),ROOT/'policies/llm_adversarial_planner.py',ROOT/'policies/intent_validation.py',
        ROOT/'policies/skills_planner.py',ROOT/'policies/rag_runtime.py',ROOT/'policies/rag_retrieval.py',
        ROOT/'work/skills_temporal_candidates_20261029/temporal_intents.py']
    hashes={str(p):sha(p) for p in watched};checks=Counter();failures=[];records=[]
    def check(kind,passed,detail):
        checks[kind]+=1
        if not passed:failures.append(dict(check=kind,detail=detail))
    base=LLMAdversarialPlanner._validate_attack_plan;guard=guarded_fractional_validator(base,[])
    normalize=lambda state:LLMAdversarialPlanner._normalize_env_state(None,state)
    validators={'default_integer':NS(_validate_attack_plan=base,_normalize_env_state=normalize),
        'optional_fractional':NS(_validate_attack_plan=guard,_normalize_env_state=normalize)}
    transforms=[('translate',0.,np.array([17.,-31.])),('rotate',.8,np.zeros(2)),
        ('quarter_turn',np.pi/2,np.array([-40.,80.])),('opposite_turn',-np.pi+.13,np.array([1000.,-2000.]))]
    for item in inputs['records']:
        key=item['scene'],item['step'];state=item['state'];h=historical[key];catalogs={'original_supplied':item['original_context']['candidates'],
            'historical_unified':h['modes']['False']['candidates'],'historical_expanded':h['modes']['True']['candidates']}
        row=dict(scene=key[0],step=key[1],source_actor_count=len(state['agents']),catalogs={},original_default_rejections=[])
        for name,candidates in catalogs.items():
            row['catalogs'][name]=dict(count=len(candidates),decisions=[])
            for candidate in candidates:
                decisions={kind:validate_world_intent(planner,state,candidate) for kind,planner in validators.items()}
                if name!='original_supplied':check('historical_published_default_recheck',decisions['default_integer']['accepted'],dict(key=key,catalog=name,candidate=candidate['candidate_id']))
                elif not decisions['default_integer']['accepted']:row['original_default_rejections'].append(dict(candidate_id=candidate['candidate_id'],**decisions['default_integer']))
                row['catalogs'][name]['decisions'].append(dict(candidate_id=candidate['candidate_id'],**decisions))
                for label,angle,shift in transforms:
                    moved,rot=transform_state(state,angle,shift);changed=dict(candidate,anchors=(np.array(candidate['anchors'])@rot.T+shift).tolist())
                    for kind,planner in validators.items():
                        decision=validate_world_intent(planner,moved,changed);expected=decisions[kind]
                        check('rigid_'+kind+'_decision',decision['accepted']==expected['accepted'],dict(key=key,catalog=name,transform=label,candidate=candidate['candidate_id']))
                        for metric,value in expected['metrics'].items():
                            drift=float(np.max(np.abs(np.array(decision['metrics'][metric])-np.array(value))))
                            check('rigid_'+kind+'_metrics',drift<1e-8,dict(key=key,metric=metric,drift=drift))
        candidates=catalogs['historical_expanded'];context=dict(candidates=candidates)
        query=current_facts(state,context,'QA_policy','QA_fingerprint','QA_protocol',item['step'],.5)
        # 仅内存中的检索器单元夹具；不是新攻击案例、模型输出或有效风险测量。
        fake=dict(case_id='QA_only_closed_case',status='closed',policy_id='QA_policy',policy_key='QA_fingerprint',protocol_key='QA_protocol',
            scene_conditions=query['scene_conditions'],query_facts=copy.deepcopy(query),D=None,measurement_valid=False,is_failure=True)
        snapshot=dict(records=[fake],revision=1);retriever=CaseRetriever();baseline=retriever.retrieve(snapshot,query)
        for label,angle,shift in transforms:
            moved,_=transform_state(state,angle,shift);changed=current_facts(moved,context,'QA_policy','QA_fingerprint','QA_protocol',item['step'],.5)
            check('RAG_current_fact_rigid_invariance',compare_facts(query,changed),dict(key=key,transform=label))
            found=retriever.retrieve(snapshot,changed)
            check('RAG_retrieved_ID_rigid_invariance',found['audit']['selected_case_ids']==baseline['audit']['selected_case_ids'],dict(key=key,transform=label))
            for a,b in zip(found['retrieved_documents'],baseline['retrieved_documents']):
                check('RAG_relevance_score_rigid_invariance',abs(a['current_fact_similarity']-b['current_fact_similarity'])<1e-8,dict(key=key,transform=label))
        catalog={c['candidate_id']:c for c in candidates};ids=list(catalog)[:6]
        response=dict(request_id='QA_only',state_hash=canonical_hash(state),catalog_hash=canonical_hash(catalog),ranking=ids,
            scene_analysis='Deterministic fixture, no LLM call',candidate_reasons={cid:'fixture, not semantic gold' for cid in ids},no_attack_reason='' if ids else 'empty fixture catalog')
        SkillsAttackPlanner.validate_response(response,'QA_only',response['state_hash'],response['catalog_hash'],catalog,False)
        check('schema_positive',True,dict(key=key))
        bads=[dict(response,request_id='wrong'),dict(response,state_hash='wrong'),dict(response,catalog_hash='wrong'),
            dict(response,ranking=['invented']),dict(response,controls=[]),dict(response,inner_lr=.1)]
        if ids:bads.append(dict(response,ranking=[ids[0],ids[0]]))
        for bad in bads:
            try:SkillsAttackPlanner.validate_response(bad,'QA_only',response['state_hash'],response['catalog_hash'],catalog,False);rejected=False
            except ValueError:rejected=True
            check('schema_negative_identity_ID_boundary',rejected,dict(key=key))
        records.append(row)
    assert all(sha(p)==expected for p,expected in hashes.items())
    summary=dict(original_input_count=40,original_scenes=len(inputs['source_traces']),records_checked=len(records),
        original_empty_catalogs=sum(r['catalogs']['original_supplied']['count']==0 for r in records),
        original_nonempty_catalogs=sum(r['catalogs']['original_supplied']['count']>0 for r in records),
        original_default_rejected_candidates=sum(len(r['original_default_rejections']) for r in records),
        total_checks=sum(checks.values()),checks_by_kind=dict(checks),failure_count=len(failures),passed=not failures,
        API_requests=0,new_diffusion_samples=0,new_execution_frames=0,D=None,source_unchanged=True,
        scope='original40_historical_catalog_recheck_and_current_pure_functions; not_rebuilt_latest_catalogs_or_LLM_performance',
        free_prose_correctness_certified=False,whole_goal_complete=False)
    args.output.mkdir(parents=True)
    (args.output/'verification.json').write_text(json.dumps(dict(summary=summary,records=records,failures=failures),ensure_ascii=False,indent=2))
    (args.output/'source_sha256.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(summary,ensure_ascii=False))
    if failures:sys.exit(2)


if __name__=='__main__':main()
