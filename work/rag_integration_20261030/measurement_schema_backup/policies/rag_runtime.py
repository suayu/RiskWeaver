"""Skills/RAG实验、规划事件与真实执行的公共生命周期；不修改风险定义或连续门控。"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import copy,hashlib,json,math,re,time,uuid

from policies.rag_store import AttackRAGStore,canonical,digest
from policies.rag_retrieval import CaseRetriever,finite
from policies.ego_profile import clean_json,policy_id_for_sim


def plain(value):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):return OmegaConf.to_container(value,resolve=True)
    except ImportError:pass
    if isinstance(value,SimpleNamespace):return {k:plain(v) for k,v in vars(value).items()}
    return clean_json(value)


def initialize_rag_experiment(sim_cfg,directory):
    """由真正实验入口调用一次；new总是新文件，attach仅用于父启动器已新建的共享文件。"""
    settings=getattr(sim_cfg,'rag',None)
    if settings is None or not (bool(getattr(settings,'enabled',False)) or bool(getattr(settings,'record_events',False))):return None
    mode=str(getattr(settings,'knowledge_mode','new'))
    if mode=='new':
        store=AttackRAGStore.create_new(Path(directory)/'rag_knowledge',
            inherit=bool(getattr(settings,'inherit',False)),source_path=getattr(settings,'inherit_path',None))
    elif mode=='attach':
        if bool(getattr(settings,'inherit',False)):raise ValueError('RAG_attach_does_not_clone_again')
        store=AttackRAGStore(getattr(settings,'knowledge_path',None))
        expected=getattr(settings,'attach_experiment_id',None)
        if not expected or store.snapshot()['current_experiment_id']!=expected:raise ValueError('RAG_attach_requires_parent_experiment_identity')
    else:raise ValueError('unknown_RAG_knowledge_mode')
    document=store.snapshot()
    try:
        from omegaconf import OmegaConf,open_dict
        if OmegaConf.is_config(settings):
            with open_dict(settings):
                settings.runtime_knowledge_path=str(store.path.resolve());settings.runtime_experiment_id=document['current_experiment_id']
        else:raise ImportError
    except ImportError:
        settings.runtime_knowledge_path=str(store.path.resolve());settings.runtime_experiment_id=document['current_experiment_id']
    receipt=dict(knowledge_path=str(store.path.resolve()),experiment_id=document['current_experiment_id'],
        initial_sha256=hashlib.sha256(store.path.read_bytes()).hexdigest(),initial_records=len(document['records']),
        retrieval_enabled=bool(getattr(settings,'enabled',False)),record_events=True,
        lineage=document['experiments'][-1],rules_are_not_retrieved=True)
    Path(directory).mkdir(parents=True,exist_ok=True)
    target=Path(directory)/'rag_experiment.json'
    if target.exists():raise FileExistsError('do_not_reinitialize_same_experiment_RAG_receipt')
    target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    return receipt


def current_facts(state,context,policy_id,policy_key,protocol_key,source_step,u):
    """查询只有当前几何、地图/路线及已公开候选；没有当前攻击的未来测量或LLM改写。"""
    ego=state['ego_state'];actors=[];issues=[]
    if len(ego)<5 or not all(finite(float(x)) for x in ego[:5]):raise ValueError('invalid_current_ego_for_RAG_query')
    speed=math.hypot(ego[2],ego[3]);heading=ego[4]
    for row in state.get('agents',[]):
        actor=row.get('state',[])
        if len(actor)<5 or not all(finite(float(x)) for x in actor[:5]):issues.append(row.get('id'));continue
        dx,dy=actor[0]-ego[0],actor[1]-ego[1];distance=math.hypot(dx,dy)
        right=dx*math.sin(heading)-dy*math.cos(heading);forward=dx*math.cos(heading)+dy*math.sin(heading)
        rate=(dx*(actor[2]-ego[2])+dy*(actor[3]-ego[3]))/distance if distance>0 else None
        actors.append(dict(current_target_id=row['id'],current_distance_m=distance,right_m=right,forward_m=forward,
            radial_gap_rate_mps=rate,longitudinal_relation='ahead' if forward>.5 else 'behind' if forward<-.5 else 'aligned',
            lateral_relation='right' if right>.5 else 'left' if right<-.5 else 'aligned',
            motion_direction='closing' if rate is not None and rate< -1e-6 else 'separating' if rate is not None and rate>1e-6 else 'unknown_or_stationary'))
    route=state.get('route',[]);angles=[]
    near=[p for p in route if len(p)>=2 and all(finite(float(x)) for x in p[:2]) and math.hypot(p[0]-ego[0],p[1]-ego[1])<30]
    for a,b in zip(near,near[1:]):
        if math.hypot(b[0]-a[0],b[1]-a[1])>.001:angles.append(math.atan2(b[1]-a[1],b[0]-a[0]))
    turns=[abs(math.atan2(math.sin(b-a),math.cos(b-a))) for a,b in zip(angles,angles[1:])]
    condition=dict(speed_bin='low' if speed<5 else 'medium' if speed<15 else 'high',
        density_bin='dense' if sum(a['current_distance_m']<35 for a in actors)>=3 else 'sparse',
        road_shape='curved' if turns and max(turns)>.15 else 'straight')
    candidates=context.get('candidates',[])
    return dict(policy_id=policy_id,policy_key=policy_key,protocol_key=protocol_key,source_step=int(source_step),
        scene_conditions=condition,actors=actors,
        candidate_strategies=sorted({str(c['strategy']) for c in candidates}),
        candidate_families=sorted({str(c.get('attack_family',c['strategy'])) for c in candidates}),
        numeric=dict(ego_speed_mps=speed,nearest_distance_m=min((a['current_distance_m'] for a in actors),default=None),
            minimum_radial_rate_mps=min((a['radial_gap_rate_mps'] for a in actors if a['radial_gap_rate_mps'] is not None),default=None),u=u),
        invalid_current_actor_ids=issues,current_fact_query_not_future_outcome=True)


class AttackRAGRuntime:
    @classmethod
    def attach_initialized(cls,sim_cfg):
        settings=getattr(sim_cfg,'rag',None)
        if settings is None:return None
        path=getattr(settings,'runtime_knowledge_path',None)
        if not path:
            if bool(getattr(settings,'enabled',False)):raise ValueError('RAG_experiment_must_be_initialized_before_generator')
            return None  # 直接组件测试/旧入口不创建文件；真正实验入口负责初始化。
        return cls(sim_cfg,AttackRAGStore(path))

    def __init__(self,sim_cfg,store):
        self.cfg=sim_cfg;self.settings=sim_cfg.rag;self.store=store;self.enabled=bool(getattr(self.settings,'enabled',False))
        if self.enabled and not bool(getattr(getattr(getattr(sim_cfg,'llm',None),'skills',None),'enabled',False)):
            raise ValueError('RAG_requires_skills_for_current_validated_pipeline')
        self.policy_id=policy_id_for_sim(sim_cfg)
        policy=str(getattr(sim_cfg,'policy','unknown'));implementation=Path(__file__).with_name('idm_policy.py' if policy=='idm' else 'rl_policy.py')
        checkpoint=getattr(self.settings,'policy_checkpoint_path',None)
        if checkpoint is None and policy=='rl':
            checkpoint=Path(str(getattr(sim_cfg,'rl_model_path','')))/(str(getattr(sim_cfg,'rl_model_name',''))+'.pt')
        checkpoint_sha=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint and Path(checkpoint).is_file() else None
        self.identity=dict(policy_id=self.policy_id,policy_name=policy,declared_version=getattr(self.settings,'policy_version',None),
            implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest() if implementation.exists() else None,
            checkpoint_sha256=checkpoint_sha,identity_scope='checkpoint_and_config' if checkpoint_sha else 'declared_policy_and_implementation_config_only')
        self.policy_key=digest(self.identity);self.experiment_id=store.snapshot()['current_experiment_id']
        evaluation=getattr(sim_cfg,'evaluation',SimpleNamespace())
        traffic=getattr(sim_cfg,'traffic_model',SimpleNamespace());guidance=getattr(traffic,'guidance',SimpleNamespace())
        self.protocol_base=dict(composite=plain(getattr(evaluation,'composite',None)),reachability=plain(getattr(evaluation,'reachability',None)),
            guidance_and_safety=plain(guidance),
            ea_protocol='ea_empty_set_zero_v2',metric_protocol_id=getattr(self.settings,'metric_protocol_id','riskweaver_current'))
        self.retriever=CaseRetriever(int(getattr(self.settings,'top_k',4)),int(getattr(self.settings,'max_context_chars',12000)),float(getattr(self.settings,'min_score',.15)))
        self.pending=None;self.episode_id=None;self.directory=None;self.retrieval_queries=0;self.retrieved_cases=0;self.written=0
        self.unattributed_failures=[];self.last_step=0
        self.storage_errors=[];self.uncommitted_records=[]
        self.current_evidence=None

    def bind_planner(self,engine):
        """在所有既有提示装饰器外绑定；保证压缩视图不会丢掉检索证据或固定规则。"""
        if getattr(engine,'_rag_bound_runtime',None) is self:return
        if getattr(engine,'_rag_bound_runtime',None) is not None:raise ValueError('planner_bound_to_other_RAG_experiment')
        engine._rag_bound_runtime=self
        engine.rag_runtime=self;plan=engine.plan
        def controlled(planner,*args,**kwargs):
            with self.prompt_bridge(planner):return plan(planner,*args,**kwargs)
        engine.plan=controlled

    @contextmanager
    def prompt_bridge(self,planner):
        request=planner._request_attack_plan;receipts=[]
        def outgoing(prompt,scene_image=None):
            prefix,raw=prompt.rsplit('\n',1);payload=json.loads(raw)
            knowledge=payload['knowledge'];present={d['path'] for d in knowledge['documents']}
            knowledge['documents'] += [d for d in self.rules_documents() if d['path'] not in present]
            if self.enabled:
                if self.current_evidence is None:raise RuntimeError('RAG_evidence_not_frozen_before_SDK_call')
                payload['context']['retrieved_attack_evidence']=copy.deepcopy(self.current_evidence)
            else:payload['context'].pop('retrieved_attack_evidence',None)
            receipts.append(dict(retrieval_enabled=self.enabled,case_ids=[d['case_id'] for d in (self.current_evidence or {}).get('documents',[])],
                fixed_rule_sha256=[d['sha256'] for d in self.rules_documents()]))
            return request(prefix+'\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False),scene_image=scene_image)
        planner._request_attack_plan=outgoing
        try:yield
        finally:
            planner._request_attack_plan=request
            if getattr(planner,'last_trace',None) is not None:planner.last_trace['rag_actual_wire']=receipts

    def commit(self,row):
        """落盘失败保留完整未提交事件，并使实验显式失败；不变成正常无攻击。"""
        try:self.store.append(clean_json(row));self.written+=1
        except (OSError,ValueError) as error:
            self.storage_errors.append(dict(stage='case_append',error_type=type(error).__name__))
            self.uncommitted_records.append(clean_json(row))

    def begin_attempt(self,env,directory,scenario_source=None,metric_source='attack_window'):
        if self.pending is not None:self.finish_window(failure='attempt_replaced_before_finalization')
        self.episode_id=uuid.uuid4().hex;self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.metric_source=metric_source
        self.protocol_key=digest(dict(self.protocol_base,metric_source=metric_source))
        self.scene_id=str(getattr(env,'current_scene_id','unknown'));self.scenario_source=scenario_source
        self.last_step=int(env.current_step)
        self.attempt_counters=(self.written,self.retrieval_queries,self.retrieved_cases)

    def begin_event(self,env,state,instruction,u):
        if self.directory is None:raise ValueError('RAG_attempt_must_begin_before_planning')
        if self.pending is not None:self.finish_window(failure='planning_replaced_unfinished_window')
        self.pending=dict(case_id=uuid.uuid4().hex,record_type='attack_event',status='open',
            created_at=time.time(),experiment_id=self.experiment_id,episode_id=self.episode_id,
            policy_id=self.policy_id,policy_key=self.policy_key,policy_identity=self.identity,protocol_key=self.protocol_key,
            source_step=int(env.current_step),scene_id=self.scene_id,scenario_source=self.scenario_source,
            instruction=list(instruction or []),u=float(u),profile_version=None,strategy=None,attack_family=None,target_id=None,
            requested_attack=False,accepted_intent=False,request_id=None,actual_attack_execution_frames=0,
            actual_obstacle_exposure_frames=0,observed_frames=0,background_safe=None,collision=None,
            generation_failure_reason=None,runtime_failure_reason=None,validation_rejections=[],measurement_records=[])
        self.pending['query_facts']=current_facts(state,{},self.policy_id,self.policy_key,self.protocol_key,env.current_step,u)
        self.current_evidence=None

    def prepare_context(self,state,context,env):
        """两组均清除旧例表直塞；开启RAG才调用检索器，候选几何和顺序不变。"""
        if self.pending is None:raise ValueError('RAG_query_without_event')
        if self.storage_errors:raise RuntimeError('RAG_storage_failed_experiment_cannot_continue_as_normal')
        result=copy.deepcopy(context);result.pop('history',None)
        for candidate in result.get('candidates',[]):candidate.pop('vulnerability',None)
        query=current_facts(state,result,self.policy_id,self.policy_key,self.protocol_key,env.current_step,self.pending['u'])
        query.update(current_case_id=self.pending['case_id'],episode_id=self.episode_id,experiment_id=self.experiment_id)
        self.pending['query_facts']=query;self.pending['scene_conditions']=query['scene_conditions']
        self.pending['profile_version']=(result.get('profile') or {}).get('version')
        if self.enabled:
            evidence=self.retriever.retrieve(self.store.snapshot(),query);self.retrieval_queries+=1
            brief=dict(query_basis={k:query[k] for k in ['policy_id','policy_key','protocol_key','source_step','scene_conditions','numeric']},
                snapshot_revision=evidence['snapshot_revision'],documents=copy.deepcopy(evidence['retrieved_documents']),
                observed_history_not_rules_or_causal_gain=True)
            removed=[]
            while brief['documents'] and len(json.dumps(brief,ensure_ascii=False))>self.retriever.max_chars:
                removed.append(brief['documents'].pop()['case_id'])
            evidence['audit']['wire_budget_excluded']=removed
            self.current_evidence=brief
            self.retrieved_cases+=len(brief['documents']);result['retrieved_attack_evidence']=brief
        else:evidence=dict(query_facts=query,retrieved_documents=[],audit=dict(reason='RAG_disabled_no_retrieval_called'))
        self.pending['retrieved_case_ids']=[d['case_id'] for d in (self.current_evidence or {}).get('documents',[])]
        if self.directory:
            with (self.directory/'rag_queries.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(clean_json(evidence),ensure_ascii=False,allow_nan=False)+'\n')
        return result

    def rules_documents(self):
        root=Path(__file__).resolve().parents[1]/'skills/riskweaver'
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));documents=[]
        for name in ['rw-current-fact-reasoning/SKILL.md','rw-case-evidence/SKILL.md']:
            raw=(root/name).read_bytes();sha=hashlib.sha256(raw).hexdigest()
            if manifest['files'].get(name)!=sha:raise ValueError('RAG_skills_rule_hash_mismatch')
            documents.append(dict(path=name,sha256=sha,text=raw.decode('utf-8')))
        return documents

    def planned(self,plan,trace,service_failure=False):
        if self.pending is None:return
        missing=plan is None
        row=self.pending;plan=plan or {};trace=trace or {};response=trace.get('response_summary') or {}
        row.update(requested_attack=bool(plan.get('attack')),request_id=plan.get('request_id') or trace.get('request_id'),
            strategy=plan.get('strategy'),attack_family=plan.get('attack_family'),target_id=plan.get('attack_target_id'),
            candidate_id=plan.get('candidate_id'),anchors=copy.deepcopy(plan.get('anchors',[])),duration_frames=plan.get('duration',10),
            planner_reason=plan.get('reason'),planner_opinions=copy.deepcopy(response),
            planner_validation=trace.get('plan_validation'),planner_model=trace.get('successful_model'),
            skill_manifest=copy.deepcopy(trace.get('skill_manifest')),
            planned_obstacles=copy.deepcopy(plan.get('obstacle_plan',[])))
        row['validation_rejections']=[dict(reason=r.get('reason')) for r in trace.get('schema_rejections',[])]+copy.deepcopy(trace.get('compile_attempts',[]))
        if service_failure:row['generation_failure_reason']='planner_service_failure'
        elif trace.get('plan_validation')=='plan_validation_failed':row['generation_failure_reason']=trace.get('validation_reason') or 'plan_validation_failed'
        elif missing:row['generation_failure_reason']=trace.get('validation_reason') or 'no_valid_highlevel_plan'
        cited=set(re.findall(r'\[case:([A-Za-z0-9_-]+)\]',canonical(response)))
        supplied=set(row.get('retrieved_case_ids',[]))
        row['case_citation_audit']=dict(cited_case_ids=sorted(cited),unknown_case_ids=sorted(cited-supplied),
            citation_presence_is_not_semantic_support_or_benefit_certificate=True)

    def mark_accepted(self):
        if self.pending:self.pending['accepted_intent']=True

    def mark_obstacles(self,created):
        if self.pending:self.pending['obstacle_created_ids']=[str(o['id']) for o in created]

    def observe(self,env,audit,info,trace,measurement=None):
        self.last_step=int(env.current_step)
        if self.pending is None:return
        row=self.pending;row['observed_frames']+=1;row['collision']=bool(row.get('collision')) or bool(info.get('collision'))
        safe=not bool(trace.previous_background or trace.previous_static);row['background_safe']=safe if row['background_safe'] is None else row['background_safe'] and safe
        if audit and audit.get('executed') and audit.get('target_id')==row['target_id'] and audit.get('request_id')==row['request_id']:
            row['actual_attack_execution_frames']+=1
        if set(row.get('obstacle_created_ids',[]))&{str(o['id']) for o in env.get_static_obstacles()}:row['actual_obstacle_exposure_frames']+=1
        if measurement is not None:row['measurement_records'].append(clean_json(copy.deepcopy(measurement)))

    def note_failure(self,reason,runtime=False):
        if self.pending is not None:self.pending['runtime_failure_reason' if runtime else 'generation_failure_reason']=str(reason)
        else:self.unattributed_failures.append(dict(reason=str(reason),runtime=bool(runtime),source_step=self.last_step))

    def finish_window(self,window=None,events=(),failure=None):
        if self.pending is None:return
        row=copy.deepcopy(self.pending);measurements=row.pop('measurement_records',[])
        if failure:row['generation_failure_reason']=str(failure)
        if self.metric_source=='e1_formal_per_transition':
            valid=[m for m in measurements if m.get('valid') and finite(m.get('difficulty'))]
            weight=sum(m.get('weight',1) for m in valid)
            row['D']=sum(m.get('weight',1)*m['difficulty'] for m in valid)/weight if weight else None
            row['valid_observation_frames']=weight;row['measurement_valid']=bool(weight)
            ea=[m.get('ea_mps2') for m in measurements if finite(m.get('ea_mps2'))]
            row['metrics']=dict(mean_evasive_acceleration_mps2=sum(ea)/len(ea) if ea else None,
                minimum_ttc_s=min((m.get('metrics',{}).get('avg_min_ttc') for m in valid if finite((m.get('metrics') or {}).get('avg_min_ttc'))),default=None))
            events=[m['reachability'] for m in measurements if isinstance(m.get('reachability'),dict)]
        else:
            valid=bool(window and window.get('valid') and finite(window.get('difficulty')))
            row['D']=window['difficulty'] if valid else None;row['measurement_valid']=valid
            row['valid_observation_frames']=window.get('weight',0) if valid else 0
            row['metrics']=copy.deepcopy((window or {}).get('metrics',{}))
        usable=[e for e in events if finite(e.get('original_area_m2')) and e['original_area_m2']>0 and finite(e.get('dangerous_area_m2'))]
        row['drivable_area_change_ratio']=sum((e['dangerous_area_m2']-e['original_area_m2'])/e['original_area_m2'] for e in usable)/len(usable) if usable else None
        exposure=row['actual_attack_execution_frames'] or row['actual_obstacle_exposure_frames']
        if row['requested_attack'] and not exposure and not row['generation_failure_reason']:
            row['generation_failure_reason']='no_attack_execution' if row['accepted_intent'] else 'intent_not_applied'
        row['is_failure']=bool(row['generation_failure_reason'] or row['runtime_failure_reason'])
        row['benefit_eligible']=bool(exposure and row['measurement_valid'] and row['background_safe'] is True and not row['is_failure'])
        row['policy_conditioned_evidence']=dict(policy_key=self.policy_key,protocol_key=self.protocol_key,
            scope='observed_outcome_under_ego_policy_not_causal_attack_uplift',D=row['D'],metrics=row['metrics'],
            actual_attack_execution_frames=row['actual_attack_execution_frames'],
            drivable_area_change_ratio=row['drivable_area_change_ratio'],collision=row['collision'],
            background_safe=row['background_safe'],causal_attack_uplift_certified=False)
        row.update(status='closed',closed_at=time.time(),metric_source=self.metric_source,
            scene_conditions=row.get('scene_conditions',row['query_facts']['scene_conditions']))
        self.commit(row);self.pending=None

    def finish_attempt(self,failure=None):
        self.finish_window(failure=failure)
        for index,item in enumerate(self.unattributed_failures):
            # 窗口之外的失败独立存档，不倒填进此前已完成攻击，也不冒充高层规划失败。
            row=dict(case_id=digest([self.experiment_id,self.episode_id,'unattributed_failure',index,item]),record_type='attack_event',status='closed',
                experiment_id=self.experiment_id,episode_id=self.episode_id,policy_id=self.policy_id,policy_key=self.policy_key,
                policy_identity=self.identity,protocol_key=self.protocol_key,source_step=item['source_step'],
                scene_id=self.scene_id,scenario_source=self.scenario_source,query_facts={},scene_conditions={},strategy=None,
                actual_attack_execution_frames=0,D=None,measurement_valid=False,valid_observation_frames=0,
                is_failure=True,benefit_eligible=False,unattributed_to_attack=True,
                generation_failure_reason=None if item['runtime'] else item['reason'],runtime_failure_reason=item['reason'] if item['runtime'] else None,
                policy_conditioned_evidence=None)
            self.commit(row)
        self.unattributed_failures=[]
        try:
            document=self.store.snapshot();knowledge_sha=hashlib.sha256(self.store.path.read_bytes()).hexdigest()
        except (OSError,ValueError) as error:
            self.storage_errors.append(dict(stage='final_snapshot',error_type=type(error).__name__))
            document={'records':[]};knowledge_sha=None
        start=getattr(self,'attempt_counters',(0,0,0))
        return dict(knowledge_path=str(self.store.path),knowledge_sha256=knowledge_sha,
            experiment_id=self.experiment_id,retrieval_enabled=self.enabled,records_written=self.written,
            attempt_records_written=self.written-start[0],retrieval_queries=self.retrieval_queries-start[1],
            retrieved_case_count=self.retrieved_cases-start[2],total_records=len(document['records']),
            storage_failure=bool(self.storage_errors),storage_errors=self.storage_errors,
            uncommitted_records=self.uncommitted_records)
