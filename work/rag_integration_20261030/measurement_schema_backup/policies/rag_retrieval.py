"""当前事实驱动的结构/BM25案例检索；规则不入库，收益严格按自车策略和协议隔离。"""
from collections import Counter
import copy,math,re


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def tokens(facts,strategy=None):
    """查询只用结构化事实标签，不调用LLM改写或读取未来执行结果。"""
    result=[]
    for key,value in facts.get('scene_conditions',{}).items():result += [key,str(value)]
    for item in facts.get('actors',[]):
        result += [str(item.get(k)) for k in ['longitudinal_relation','lateral_relation','motion_direction'] if item.get(k) is not None]
    result += list(facts.get('candidate_strategies',[]))+list(facts.get('candidate_families',[]))
    if strategy:result.append(str(strategy))
    return [x.lower() for s in result for x in re.findall(r'[\w.-]+',s)]


def numeric_similarity(a,b):
    scales={'ego_speed_mps':5.,'nearest_distance_m':10.,'minimum_radial_rate_mps':5.,'u':.25}
    values=[math.exp(-abs(a[k]-b[k])/scale) for k,scale in scales.items() if finite(a.get(k)) and finite(b.get(k))]
    return sum(values)/len(values) if values else None


def structural_similarity(query,case):
    qc=query.get('scene_conditions',{});cc=case.get('scene_conditions',{})
    values=[float(qc[k]==cc[k]) for k in qc.keys()&cc.keys()]
    q=set(query.get('candidate_families',[]));c=set(case.get('candidate_families',[]))
    if q and c:values.append(len(q&c)/len(q|c))
    return sum(values)/len(values) if values else 0.


def bm25_scores(query,documents):
    """固定k1=1.2、b=.75；只作为相关性，绝不冒称攻击成功概率。"""
    if not documents:return []
    counts=[Counter(doc) for doc in documents];average=sum(map(len,documents))/len(documents) or 1.
    frequencies=Counter(term for doc in counts for term in doc)
    scores=[]
    for doc,terms in zip(documents,counts):
        score=0.
        for term in set(query):
            tf=terms.get(term,0)
            if not tf:continue
            df=frequencies[term];idf=math.log(1+(len(documents)-df+.5)/(df+.5))
            score += idf*tf*2.2/(tf+1.2*(.25+.75*len(doc)/average))
        scores.append(score)
    return scores


class CaseRetriever:
    def __init__(self,top_k=4,max_context_chars=12000,min_score=.15,allow_legacy_cases=True):
        if not 1<=top_k<=12 or max_context_chars<1500 or not 0<=min_score<=1:raise ValueError('invalid_RAG_retrieval_budget')
        self.top_k=top_k;self.max_chars=max_context_chars;self.min_score=min_score;self.allow_legacy=allow_legacy_cases

    def retrieve(self,snapshot,query):
        """保留检索排除原因；不从知识库删除不利、失败或未知测量事件。"""
        eligible=[];excluded=Counter()
        for row in snapshot['records']:
            reason=None;legacy=row.get('source')=='legacy_profile_knowledge'
            if row.get('status')!='closed':reason='unfinished_event'
            elif row.get('case_id')==query.get('current_case_id'):reason='current_event'
            elif row.get('policy_id')!=query['policy_id']:reason='different_ego_policy'
            elif row.get('episode_id')==query.get('episode_id') and row.get('experiment_id')==query.get('experiment_id') and row.get('source_step',-1)>=query['source_step']:
                reason='same_episode_future_or_current_step'
            elif legacy:
                if not self.allow_legacy:reason='legacy_identity_unverified'
            elif row.get('policy_key')!=query['policy_key']:reason='different_policy_fingerprint'
            elif row.get('protocol_key')!=query['protocol_key']:reason='different_metric_or_safety_protocol'
            if reason:excluded[reason]+=1;continue
            facts=copy.deepcopy(row.get('query_facts') or dict(scene_conditions=row.get('scene_conditions',{}),candidate_families=[row.get('strategy')]))
            eligible.append((row,facts,'legacy_policy_id_only' if legacy else 'same_policy_fingerprint_and_protocol'))
        documents=[tokens(facts,row.get('strategy')) for row,facts,_ in eligible]
        bm25=bm25_scores(tokens(query),documents);maximum=max(bm25,default=0.)
        candidates=[]
        for (row,facts,compatibility),lexical in zip(eligible,bm25):
            structure=structural_similarity(query,facts);numeric=numeric_similarity(query.get('numeric',{}),facts.get('numeric',{}))
            normalized=lexical/maximum if maximum else 0.
            score=.45*structure+.35*normalized+.2*(numeric if numeric is not None else 0.)
            if score<self.min_score:excluded['low_current_fact_relevance']+=1;continue
            candidates.append(dict(record=row,score=score,compatibility=compatibility,
                score_components=dict(structure=structure,bm25=lexical,numeric=numeric)))
        candidates.sort(key=lambda r:(-r['score'],r['record']['case_id']))
        selected=candidates[:self.top_k]
        # 有相关反例时保留一条，避免历史高D把失败经验挤掉；不是筛除实验失败。
        failure=next((r for r in candidates if r['record'].get('is_failure')),None)
        if self.top_k>=2 and failure and not any(r['record'].get('is_failure') for r in selected):
            selected=selected[:self.top_k-1]+[failure]
        documents=[];used=0;budget_excluded=[]
        import json
        for candidate in selected:
            row=candidate['record'];same=candidate['compatibility']=='same_policy_fingerprint_and_protocol'
            observed=dict(D=row.get('D') if row.get('measurement_valid') else None,
                measurement_valid=row.get('measurement_valid',False),
                valid_observation_frames=row.get('valid_observation_frames'),
                actual_attack_execution_frames=row.get('actual_attack_execution_frames',0),
                metrics=row.get('metrics',{}),drivable_area_change_ratio=row.get('drivable_area_change_ratio'),
                background_safe=row.get('background_safe'),collision=row.get('collision'))
            benefit=bool(same and row.get('benefit_eligible'))
            document=dict(case_id=row['case_id'],kind='failure_counterexample' if row.get('is_failure') else 'attack_event_case',
                policy_compatibility=candidate['compatibility'],scene_conditions=row.get('scene_conditions',{}),
                current_fact_similarity=candidate['score'],similarity_is_not_success_probability=True,
                strategy=row.get('strategy'),attack_family=row.get('attack_family'),source_step=row.get('source_step'),
                failure_reason=row.get('generation_failure_reason'),runtime_failure_reason=row.get('runtime_failure_reason'),
                validation_rejections=row.get('validation_rejections',[]),
                observed_outcome=observed,
                policy_conditioned_benefit_evidence=copy.deepcopy(row.get('policy_conditioned_evidence')) if benefit else None,
                benefit_evidence_eligible=benefit,causal_attack_uplift_certified=False,
                historical_role_only_no_current_vehicle_ID_or_anchor_replay=True,
                planner_opinion_excerpt=str(row.get('planner_reason') or '')[:500],
                excerpt_is_untrusted_historical_opinion_not_rule=True)
            size=len(json.dumps(document,ensure_ascii=False,allow_nan=False))
            if used+size>self.max_chars:budget_excluded.append(row['case_id']);continue
            documents.append(document);used+=size
        return dict(query_facts=copy.deepcopy(query),snapshot_revision=snapshot.get('revision',0),
            retrieved_documents=documents,audit=dict(total_records=len(snapshot['records']),eligible_records=len(eligible),
                excluded=dict(excluded),ranked_case_ids=[c['record']['case_id'] for c in candidates],
                selected_case_ids=[d['case_id'] for d in documents],context_characters=used,
                context_budget_excluded=budget_excluded,top_k=self.top_k,
                scorer='fixed_structure_.45+normalized_BM25_.35+numeric_.20',
                no_LLM_query_rewrite=True,all_data_from_current_experiment_knowledge_file=True,
                original_records_not_deleted=True))
