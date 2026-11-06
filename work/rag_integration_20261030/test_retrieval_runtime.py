"""当前事实、检索隔离、真实请求边界及执行写回；不发API、不运行新仿真。"""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,importlib.util,json,sys
import pytest

HERE=Path(__file__).resolve().parent
for name in ['rag_store','rag_retrieval','rag_runtime']:
    spec=importlib.util.spec_from_file_location('policies.'+name,HERE/'source_after/policies'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
from policies.rag_runtime import AttackRAGRuntime,initialize_rag_experiment,current_facts
from policies.rag_store import AttackRAGStore
from policies.rag_retrieval import CaseRetriever


def cfg(enabled=True):
    return NS(policy='idm',llm=NS(skills=NS(enabled=True)),rag=NS(enabled=enabled,record_events=True,inherit=False,inherit_path=None))


def state():
    return dict(ego_state=[0.,0.,5.,0.,0.],agents=[dict(id=7,state=[10.,-2.,4.,0.,0.])],route=[[0,0],[10,0],[20,0]])


def context():
    return dict(profile={'version':3},history={'recent_examples':['MUST_NOT_SEND']},
        candidates=[dict(candidate_id='a',target_id=7,strategy='cut_in',attack_family='merge',vulnerability={'recent_examples':['MUST_NOT_SEND']})])


def make_runtime(tmp_path,enabled=True,metric='e1_formal_per_transition'):
    settings=cfg(enabled);receipt=initialize_rag_experiment(settings,tmp_path/'experiment')
    runtime=AttackRAGRuntime.attach_initialized(settings)
    runtime.begin_attempt(NS(current_step=0,current_scene_id=74),tmp_path/'attempt',metric_source=metric)
    return settings,receipt,runtime


def case(runtime,ident='past',**changes):
    facts=current_facts(state(),context(),runtime.policy_id,runtime.policy_key,runtime.protocol_key,0,.5)
    row=dict(case_id=ident,record_type='attack_event',status='closed',policy_id=runtime.policy_id,policy_key=runtime.policy_key,
        protocol_key=runtime.protocol_key,experiment_id='old',episode_id='old',source_step=0,query_facts=facts,
        scene_conditions=facts['scene_conditions'],strategy='cut_in',actual_attack_execution_frames=10,D=.7,
        measurement_valid=True,valid_observation_frames=8,background_safe=True,is_failure=False,benefit_eligible=True,
        policy_conditioned_evidence={'D':.7,'causal_attack_uplift_certified':False})
    row.update(changes);return row


def event(runtime,step=1):
    env=NS(current_step=step,current_scene_id=74,get_static_obstacles=lambda:[])
    runtime.begin_event(env,state(),['challenge'],.5)
    return env


def test_query_uses_current_geometry_and_motion_not_future_outcome():
    q=current_facts(state(),context(),'idm','finger','proto',11,.5)
    assert q['actors'][0]['lateral_relation']=='right' and q['actors'][0]['longitudinal_relation']=='ahead'
    assert q['actors'][0]['motion_direction']=='closing'
    assert q['numeric']['ego_speed_mps']==5 and q['source_step']==11 and 'D' not in q


def test_retrieval_filters_policy_protocol_open_current_future_and_never_ranks_reward(tmp_path):
    _,_,r=make_runtime(tmp_path);query=current_facts(state(),context(),r.policy_id,r.policy_key,r.protocol_key,5,.5)
    query.update(current_case_id='self',episode_id='now',experiment_id='now')
    rows=[case(r,'good'),case(r,'failure',is_failure=True,D=None,measurement_valid=False,benefit_eligible=False),
        case(r,'wrong-policy',policy_id='other'),case(r,'wrong-checkpoint',policy_key='other'),
        case(r,'wrong-protocol',protocol_key='other'),case(r,'open',status='open'),case(r,'self'),
        case(r,'future',episode_id='now',experiment_id='now',source_step=5)]
    result=CaseRetriever(top_k=2).retrieve({'records':rows,'revision':8},query)
    assert {d['case_id'] for d in result['retrieved_documents']}=={'good','failure'}
    assert len(result['audit']['excluded'])==6
    bad=next(d for d in result['retrieved_documents'] if d['case_id']=='failure')
    assert bad['observed_outcome']['D'] is None and bad['policy_conditioned_benefit_evidence'] is None


def test_legacy_never_becomes_verified_policy_benefit(tmp_path):
    _,_,r=make_runtime(tmp_path);q=current_facts(state(),context(),r.policy_id,r.policy_key,r.protocol_key,5,.5)
    result=r.retriever.retrieve({'records':[case(r,source='legacy_profile_knowledge',policy_key=None,protocol_key=None)],'revision':1},q)
    doc=result['retrieved_documents'][0]
    assert doc['policy_compatibility']=='legacy_policy_id_only' and doc['policy_conditioned_benefit_evidence'] is None


def test_unattributed_failure_with_no_scene_facts_does_not_break_next_query(tmp_path):
    _,_,r=make_runtime(tmp_path);q=current_facts(state(),context(),r.policy_id,r.policy_key,r.protocol_key,5,.5)
    row=case(r,'outer_failure',query_facts={},scene_conditions={},strategy=None,is_failure=True)
    result=r.retriever.retrieve({'records':[row],'revision':1},q)
    assert result['retrieved_documents']==[] and result['audit']['excluded']['low_current_fact_relevance']==1


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_sdk_boundary_preserves_cases_after_compact_transport(enabled,tmp_path):
    _,_,r=make_runtime(tmp_path,enabled);r.store.append(case(r));env=event(r)
    payload_context=r.prepare_context(state(),context(),env)
    assert 'history' not in payload_context and 'vulnerability' not in payload_context['candidates'][0]
    captured=[]
    planner=NS(last_trace={},_request_attack_plan=lambda prompt,scene_image=None:captured.append(json.loads(prompt.rsplit('\n',1)[1])))
    # 模拟旧压缩器清除未知字段；RAG必须在最终真实调用边界补回，而非只存在trace中。
    def compact_plan(planner):
        payload=dict(context={'candidates':payload_context['candidates']},knowledge={'documents':[]})
        planner._request_attack_plan('rank\n'+json.dumps(payload))
    engine=NS(plan=compact_plan);r.bind_planner(engine);engine.plan(planner)
    docs=captured[0]['knowledge']['documents'];assert len(docs)==2 and all(d['sha256'] for d in docs)
    if enabled:
        assert captured[0]['context']['retrieved_attack_evidence']['documents'][0]['case_id']=='past'
        assert r.retrieval_queries==1
    else:
        assert 'retrieved_attack_evidence' not in captured[0]['context'] and r.retrieval_queries==0
    assert planner.last_trace['rag_actual_wire'][0]['retrieval_enabled']==enabled


def test_no_rag_does_not_even_call_retriever(tmp_path):
    _,_,r=make_runtime(tmp_path,False);env=event(r)
    r.retriever.retrieve=lambda *_:pytest.fail('OFF must never read cases')
    r.prepare_context(state(),context(),env)
    assert r.pending['retrieved_case_ids']==[]


def test_execution_id_attribution_and_frame_weighted_D(tmp_path):
    _,_,r=make_runtime(tmp_path);env=event(r)
    r.prepare_context(state(),context(),env)
    r.planned(dict(attack=True,request_id='req',attack_target_id=7,strategy='cut_in',anchors=[[1,2],[3,4]],duration=10),{})
    r.mark_accepted();trace=NS(previous_background=False,previous_static=False)
    for i,(request,value,weight,valid) in enumerate([('wrong',.9,1,True),('req',.3,3,True),('req',None,1,False)]):
        env.current_step=i+2
        r.observe(env,dict(executed=True,request_id=request,target_id=7),{},trace,
            dict(valid=valid,difficulty=value,weight=weight,ea_mps2=0.,metrics={},reachability={'original_area_m2':10.,'dangerous_area_m2':4.}))
    r.finish_window();row=r.store.snapshot()['records'][0]
    assert row['actual_attack_execution_frames']==2 and row['valid_observation_frames']==4
    assert row['D']==pytest.approx(.45) and row['drivable_area_change_ratio']==pytest.approx(-.6)
    assert row['benefit_eligible'] and not row['policy_conditioned_evidence']['causal_attack_uplift_certified']


def test_invalid_D_no_execution_and_failures_are_retained(tmp_path):
    _,_,r=make_runtime(tmp_path);event(r)
    r.planned(dict(attack=True,attack_target_id=7,request_id='r'),{})
    r.note_failure('joint_dynamics_rejected');r.finish_window()
    first=r.store.snapshot()['records'][0]
    assert first['D'] is None and not first['benefit_eligible'] and first['is_failure']
    r.note_failure('outer_runtime_failure',runtime=True);summary=r.finish_attempt()
    assert summary['total_records']==2 and r.store.snapshot()['records'][1]['unattributed_to_attack']


def test_obstacles_and_storage_failure_are_not_silent_normal_results(tmp_path,monkeypatch):
    _,_,r=make_runtime(tmp_path);env=event(r)
    r.planned(dict(attack=False,obstacle_plan=[{'id':'work'}]),{})
    r.mark_obstacles([{'id':'work'}]);env.get_static_obstacles=lambda:[{'id':'work'}]
    r.observe(env,None,{},NS(previous_background=False,previous_static=False),dict(valid=False,difficulty=None))
    monkeypatch.setattr(r.store,'append',lambda *_:(_ for _ in ()).throw(OSError('full')))
    summary=r.finish_attempt()
    assert summary['storage_failure'] and summary['uncommitted_records'][0]['actual_obstacle_exposure_frames']==1
    assert summary['uncommitted_records'][0]['D'] is None


def test_new_experiment_and_next_episode_share_only_explicit_KB(tmp_path):
    settings,receipt,r=make_runtime(tmp_path);r.store.append(case(r));old=r.store.path.read_bytes()
    child=cfg();child.rag.inherit=True;child.rag.inherit_path=str(r.store.path)
    other=initialize_rag_experiment(child,tmp_path/'child')
    assert other['knowledge_path']!=receipt['knowledge_path'] and other['initial_records']==1
    r.begin_attempt(NS(current_step=0,current_scene_id=182),tmp_path/'next')
    assert r.store.path.read_bytes()==old and len(r.store.snapshot()['records'])==1
    with pytest.raises(FileExistsError):initialize_rag_experiment(settings,tmp_path/'experiment')
