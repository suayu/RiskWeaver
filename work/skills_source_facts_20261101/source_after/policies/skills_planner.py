"""可开关技能链路：规则候选、共同预检、单次分析排序与有界恢复。"""
import copy
import json
import time
import uuid
from types import SimpleNamespace
import numpy as np

from policies.ego_profile import CandidateBuilder,AttackKnowledge,scene_conditions,clean_json
from policies.intent_validation import canonical_hash,validate_world_intent
from policies.skill_registry import SkillRegistry


class SkillsAttackPlanner:
    def __init__(self,sim_cfg,call_budget=4):
        self.settings=sim_cfg.llm.skills
        self.registry=SkillRegistry(getattr(self.settings,'root',None),getattr(self.settings,'level','full'))
        self.builder=CandidateBuilder(getattr(sim_cfg,'ego_profile',SimpleNamespace()))
        self.expansion=bool(getattr(self.settings,'candidate_expansion_enabled',False))
        self.limit=int(getattr(self.settings,'max_candidates',24))
        self.fallback_limit=int(getattr(self.settings,'fallback_candidates',6))
        self.repairs=int(getattr(self.settings,'max_repairs',1))
        # 默认沿用整份回复严格校验；可选逐候选隔离，不放宽被选计划的任何检查。
        self.grounding_validation_mode=str(getattr(self.settings,'grounding_validation_mode','global'))
        if self.grounding_validation_mode not in ('global','candidate_local'):
            raise ValueError('unknown skills grounding validation mode')
        self.budget=int(call_budget)
        self.calls=0
        self.seen=set()
        if not 1<=self.limit<=64 or not 1<=self.fallback_limit<=self.limit or self.repairs not in (0,1):
            raise ValueError('invalid skills candidate/repair budget')
        # 默认关闭；仅技能路径补充当前事实，保留旧调用和原校验语义。
        self.source_fact_contract = None
        enabled = getattr(self.settings, 'source_fact_contract_enabled', False)
        if not isinstance(enabled, bool):
            raise ValueError('source_fact_contract_enabled_requires_boolean')
        if enabled:
            from policies.skills_source_facts import SourceFactContract
            self.source_fact_contract = SourceFactContract(self)
            # 捕获最终子类方法，使补充事实经过压缩后仍送到实际请求端。
            self.plan = self.source_fact_contract.wrap(self.plan)

    def reset_episode(self):
        """跨场景重置调用计数；拒绝响应身份仍由每次随机请求ID和快照哈希保证。"""
        self.calls=0
        self.seen.clear()

    def build_context(self,env,state,planner,profile_context=None):
        """画像关闭时绝不读取画像；目录扩展是独立开关，可做2×2消融。"""
        if profile_context is None:
            candidates,rejected=self.builder.build(env,{},AttackKnowledge('skills-unprofiled',[]))
            context=dict(profile=None,history=None,candidates=candidates,candidate_rejections=rejected,
                         candidate_audit=copy.deepcopy(self.builder.last_audit))
        else:
            context=copy.deepcopy(profile_context)
        visible={a['id'] for a in state['agents']}
        candidates=[c for c in context['candidates'] if c['target_id'] in visible]
        if self.expansion:
            candidates+=self.expanded_candidates(env,state,planner)
        accepted=[];audit=[];duplicates=set()
        for candidate in candidates:
            key=canonical_hash(dict(target=candidate['target_id'],strategy=candidate['strategy'],anchors=candidate['anchors']))
            if key in duplicates:continue
            duplicates.add(key)
            result=validate_world_intent(planner,state,candidate)
            audit.append(dict(candidate_id=candidate['candidate_id'],**result))
            if result['accepted']:
                accepted.append(dict(candidate,semantic_evidence=result['metrics']))
        # 目录上限与排序规则固定；不使用未来真实结果筛选候选。
        accepted.sort(key=lambda c:min(c['semantic_evidence']['aligned_ego_distance_m']))
        allow_dynamic=bool(getattr(planner,'allow_trajectory',True))
        context.update(candidates=accepted[:self.limit] if allow_dynamic else [],semantic_precheck=audit,
                       scene_conditions=scene_conditions(env),profile_enabled=profile_context is not None,
                       candidate_expansion_enabled=self.expansion,source_step=int(env.current_step),
                       scene_id=str(getattr(env,'current_scene_id','unknown')),allow_dynamic=allow_dynamic)
        if planner.use_obstacles:
            from policies.profile_obstacles import build_obstacle_candidates
            context['obstacle_candidates']=build_obstacle_candidates(env,planner,context['candidates'],self.builder)
        return clean_json(context)

    def expanded_candidates(self,env,state,planner):
        """增加跨向/后向机会，不设道路距离、当前速度或朝向资格门槛；仍经原背景动力学筛查。"""
        ego=np.asarray(env.ego_state,float)
        states=np.asarray(env.data_dict['agent'][-1],float)
        lanes=[np.asarray(l)[:,:2] for l in env.scenario_dict.get('lanes',[]) if len(l)>1]
        t=np.arange(4.)[:,None]
        output=[]
        # 同时考虑匀速自车与沿已知路线的3秒位置，避免只覆盖平行同向前车。
        route=np.asarray(state.get('route',[]),float)
        future=ego[:2]+3*ego[2:4]
        desired=[future]
        if route.ndim==2 and len(route)>1:
            near=int(np.argmin(np.linalg.norm(route[:,:2]-ego[:2],axis=1)))
            lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route[near:,:2],axis=0),axis=1))]
            desired.append(route[near+int(np.argmin(abs(lengths-3*np.linalg.norm(ego[2:4])))),:2])
        for target in state['agents']:
            ident=target['id'];s=states[ident]
            if not env.agent_active[ident]:continue
            variants=[('slow_down','forward_pressure',-s[2:4]/max(np.linalg.norm(s[2:4]),1e-6)*a) for a in (1.,2.) if np.linalg.norm(s[2:4])>=3*a]
            for point in desired:
                acceleration=2*(point-s[:2]-3*s[2:4])/9
                for cap in (1.,2.,3.8):
                    a=acceleration*min(1.,cap/max(np.linalg.norm(acceleration),1e-9))
                    variants.append(('lane_change','merge_conflict',a))
            for strategy,family,a in variants:
                anchors=s[:2]+t*s[2:4]+.5*t*t*a
                estimate,reason=self.builder.screen(env,ident,anchors,lanes)
                if reason:continue
                candidate=dict(candidate_id=f'x{len(output)}',target_id=int(ident),strategy=strategy,
                    attack_family=family,anchors=anchors.tolist(),duration=10,
                    profile_match=None,profile_evidence_frames=0,recipe='bounded_acceleration_sparse_intent_v1',**estimate)
                # 使用现有的语义验证；lane_change并不承诺沿车道，仅表达离散冲突意图。
                if validate_world_intent(planner,state,candidate)['accepted']:output.append(candidate)
        return output

    def plan(self,planner,env,state,instruction,profile_context=None,scene_image=None):
        started=time.monotonic();planner.last_request_failed=False;planner.last_error=None
        trace=dict(mode='skills_candidate_ranking',stage='context',api_calls_before=self.calls)
        planner.last_trace=trace
        try:
            context=self.build_context(env,state,planner,profile_context)
            if self.source_fact_contract is not None:
                context=self.source_fact_contract.prepare_context(state,context,env)
            rag=getattr(self,'rag_runtime',None)
            if rag is not None:
                try:context=rag.prepare_context(state,context,env)
                except Exception as error:
                    trace['rag_failure']={'error_type':type(error).__name__}
                    raise
            trace['context']=context
            if not context['candidates'] and not context.get('obstacle_candidates'):
                trace.update(plan_validation='no_feasible_candidates',stage='no_candidate')
                return dict(attack=False,attack_target_id=-1,strategy='none',anchors=[],reason='no_feasible_candidates')
            catalog={c['candidate_id']:c for c in context['candidates']}
            request_id=uuid.uuid4().hex
            state_hash=canonical_hash(state);catalog_hash=canonical_hash(context)
            trace.update(request_id=request_id,state_hash=state_hash,catalog_hash=catalog_hash)
            response=None;feedback=None;eligible_grounding_ids=None
            for repair in range(self.repairs+1):
                if self.calls>=self.budget:
                    trace.update(plan_validation='skills_api_budget_exhausted')
                    return dict(attack=False,attack_target_id=-1,strategy='none',anchors=[],reason='skills_api_budget_exhausted')
                knowledge=self.registry.load(planner.use_obstacles,repair=bool(repair))
                if rag is not None:
                    # 固定推理/历史证据使用规则在RAG开关两侧完全相同，不由检索选择。
                    knowledge=copy.deepcopy(knowledge)
                    present={d['path'] for d in knowledge['documents']}
                    knowledge['documents'] += [d for d in rag.rules_documents() if d['path'] not in present]
                    knowledge['version']=str(knowledge['version'])+'+fixed-rag-rules-v1'
                trace['skill_manifest']=dict(version=knowledge['version'],level=knowledge['level'],documents=[{k:v for k,v in d.items() if k!='text'} for d in knowledge['documents']])
                payload=dict(request_id=request_id,state_hash=state_hash,catalog_hash=catalog_hash,
                    context=context,instruction=instruction,knowledge=knowledge,repair_feedback=feedback)
                prompt=('Rank this closed catalog of discrete adversarial intents. Return JSON only with exactly: request_id, state_hash, catalog_hash, ranking (candidate ID list best first), scene_analysis (concise text grounded in supplied metrics), candidate_reasons (object mapping EVERY ranked ID to a concrete reason), no_attack_reason (empty string when ranking nonempty). '+
                    ('Also include obstacle_candidate_id (listed ID or null); it must be compatible with the first dynamic choice. ' if planner.use_obstacles else 'No obstacle fields. ')+
                    'Copy the three identity strings. No dense trajectories, controls or invented IDs. Rank one or more feasible probes if supplied evidence supports interaction; cold start is uncertainty, not automatic refusal. Empty ranking requires a concrete nonempty no_attack_reason. Background safety and dynamics are mandatory.\n'+json.dumps(payload,ensure_ascii=False,allow_nan=False))
                trace.setdefault('requests',[]).append(dict(repair=repair,prompt=prompt))
                self.calls+=1
                try:
                    _,response=planner._request_attack_plan(prompt,scene_image=scene_image)
                except Exception as error:
                    from policies.llm_adversarial_planner import ModelOutputError
                    if not isinstance(error,ModelOutputError):raise
                    feedback=dict(reason='invalid_json_output')
                    trace.setdefault('schema_rejections',[]).append(feedback)
                    if repair==self.repairs:raise
                    continue
                trace['requests'][-1]['response']=response
                try:
                    if self.grounding_validation_mode=='candidate_local':
                        from policies.candidate_grounding import validate_candidate_local
                        decision=validate_candidate_local(response,request_id,state_hash,catalog_hash,catalog,
                            planner.use_obstacles,self.validate_response,SkillsAttackPlanner.validate_response)
                        trace['candidate_grounding_validation']=decision
                        eligible_grounding_ids=set(decision['eligible_ids'])
                    else:
                        self.validate_response(response,request_id,state_hash,catalog_hash,catalog,planner.use_obstacles)
                    break
                except (ValueError,TypeError) as error:
                    if hasattr(error,'candidate_audit'):
                        trace.setdefault('candidate_grounding_failures',[]).append(error.candidate_audit)
                    feedback=dict(reason=str(error),previous_response=response)
                    trace.setdefault('schema_rejections',[]).append(feedback)
                    if repair==self.repairs:raise
            if request_id in self.seen or state_hash!=canonical_hash(env.get_state_for_planning()):
                raise ValueError('duplicate_or_stale_response')
            self.seen.add(request_id)
            ranking=response['ranking']
            trace.update(response_summary=response,candidate_ranking=ranking,stage='compile')
            if not ranking:
                obstacles=self.select_obstacle(response,context,[])
                trace['plan_validation']='no_attack' if not obstacles else 'accepted_obstacle_only'
                return dict(attack=False,attack_target_id=-1,strategy='none',anchors=[],reason=response['no_attack_reason'],request_id=request_id,**({'obstacle_plan':obstacles} if planner.use_obstacles else {}))
            for cid in ranking[:self.fallback_limit]:
                if eligible_grounding_ids is not None and cid not in eligible_grounding_ids:
                    reason=next((r['reason'] for r in trace['candidate_grounding_validation']['candidates'] if r['candidate_id']==cid),None)
                    trace.setdefault('compile_attempts',[]).append(dict(candidate_id=cid,accepted=False,
                        reason=reason,stage='candidate_grounding'))
                    continue
                c=catalog[cid];result=validate_world_intent(planner,state,c)
                trace.setdefault('compile_attempts',[]).append(dict(candidate_id=cid,**result))
                if not result['accepted']:continue
                try:obstacles=self.select_obstacle(response,context,[cid])
                except ValueError as error:
                    trace['compile_attempts'][-1].update(accepted=False,reason=str(error));continue
                trace.update(plan_validation='accepted',selected_candidate=cid,stage='accepted')
                return dict(attack=True,attack_target_id=c['target_id'],strategy=c['strategy'],attack_family=c['attack_family'],
                    anchors=c['anchors'],duration=c['duration'],reason=response['candidate_reasons'][cid],
                    request_id=request_id,candidate_id=cid,**({'obstacle_plan':obstacles} if planner.use_obstacles else {}))
            raise ValueError('ranked_candidates_failed_final_validation')
        except Exception as error:
            from policies.llm_adversarial_planner import LLMAdversarialPlanner
            planner.last_request_failed=LLMAdversarialPlanner._is_service_failure(error)
            planner.last_error=type(error).__name__ if planner.last_request_failed else str(error)
            trace.update(plan_validation='planner_service_failure' if planner.last_request_failed else 'plan_validation_failed',
                         validation_reason=planner.last_error,error_type=type(error).__name__)
            return None
        finally:
            trace.update(elapsed_seconds=time.monotonic()-started,api_calls_total=self.calls,
                successful_model=getattr(planner,'last_successful_model_name',None),
                attempted_models=getattr(planner,'last_attempted_model_names',[]),
                model_failures=getattr(planner,'last_model_failures',[]))

    @staticmethod
    def select_obstacle(response,context,ranking):
        if 'obstacle_candidate_id' not in response:return []
        from policies.profile_obstacles import chosen_obstacle
        return chosen_obstacle(response,context,ranking)

    @staticmethod
    def validate_response(response,request_id,state_hash,catalog_hash,catalog,obstacles):
        required={'request_id','state_hash','catalog_hash','ranking','scene_analysis','candidate_reasons','no_attack_reason'}
        if obstacles:required.add('obstacle_candidate_id')
        if not isinstance(response,dict) or set(response)!=required:raise ValueError('skills_schema_fields')
        if any(response[k]!=v for k,v in [('request_id',request_id),('state_hash',state_hash),('catalog_hash',catalog_hash)]):raise ValueError('skills_identity_mismatch')
        ranking=response['ranking'];reasons=response['candidate_reasons']
        if not isinstance(ranking,list) or any(not isinstance(x,str) for x in ranking) or len(set(ranking))!=len(ranking) or not set(ranking)<=set(catalog):raise ValueError('skills_ranking_not_catalog')
        if not isinstance(reasons,dict) or set(reasons)!=set(ranking) or any(not isinstance(v,str) or not v.strip() for v in reasons.values()):raise ValueError('skills_missing_candidate_reasons')
        if not isinstance(response['scene_analysis'],str) or not response['scene_analysis'].strip():raise ValueError('skills_missing_scene_analysis')
        reason=response['no_attack_reason']
        if not isinstance(reason,str) or (bool(ranking)==bool(reason.strip())):raise ValueError('skills_no_attack_reason')
