"""原十个复核初态：同一动态/静态skills，无RAG/有RAG；只验收高层，不执行仿真。"""
from pathlib import Path
import argparse,copy,hashlib,json,os,pickle,shutil,sys,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'work/skills_source_facts_20261101')]
import diagnostic as harness
from omegaconf import OmegaConf
from policies.hybrid_skills_planner import DynamicFirstStaticPlanner
from policies.rag_runtime import AttackRAGRuntime,initialize_rag_experiment
from policies.rag_store import AttackRAGStore

PROTOCOL=ROOT/'work/skills_hybrid_rag_highlevel_review10_protocol_20261101_r1'
OUTPUT=ROOT/'experiments/skills_hybrid_rag_highlevel_review10_20261101_r1'
SCENES=[6,28,45,102,114,143,146,185,229,232]
ARMS=['skills_no_rag','skills_rag']
CURRENT={}
ORIGINAL_ENGINE=harness.base.engine_for
ORIGINAL_PLANNER=harness.base.LLMAdversarialPlanner
harness.base.INSTRUCTION=['存在可行车辆候选时优先车辆攻击；无可用车辆候选时从已验证的静态组合目录选择道路拦截方案。自车未来只是预测假设，计划不等于实际执行。']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    PROTOCOL.mkdir(exist_ok=False)
    prior=ROOT/'work/skills_candidate_local_holdout_protocol_20261030_r1'
    samples=json.loads((prior/'inputs.json').read_text());manifest=json.loads((prior/'validation10.json').read_text())
    if [s['scene'] for s in samples]!=SCENES or manifest['original_scenario_indices']!=SCENES:
        raise ValueError('fixed_review_ten_changed')
    for sample,path,digest in zip(samples,manifest['scenario_files'],manifest['scenario_sha256']):
        if sha(path)!=digest:raise ValueError('source_pickle_changed')
        sample['scenario_path']=path;sample['scenario_sha256']=digest
    harness.base.save(PROTOCOL/'inputs.json',samples)
    cfg=OmegaConf.load(ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
    cfg.sim.llm.skills.source_fact_contract_enabled=True
    cfg.sim.llm.skills.static_fallback_enabled=True
    cfg.sim.llm.skills.grounding_validation_mode='candidate_local'
    cfg.sim.llm.attack_mode='joint';cfg.sim.llm.obstacles.enabled=False
    cfg.sim.rag.knowledge_mode='new';cfg.sim.rag.inherit=True;cfg.sim.rag.record_events=True
    cfg.sim.rag.runtime_knowledge_path=None;cfg.sim.rag.runtime_experiment_id=None
    seed=next((ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_rag/rag_knowledge').glob('*.json'))
    frozen=PROTOCOL/'inherit_seed.json';frozen.write_bytes(seed.read_bytes());cfg.sim.rag.inherit_path=str(frozen)
    inherited=AttackRAGStore(frozen).snapshot()['records']
    if {str(row['scene_id']) for row in inherited}&{str(s['scene_id']) for s in samples}:
        raise ValueError('inheritance_contains_current_evaluation_scene')
    OmegaConf.save(cfg,PROTOCOL/'resolved_config.yaml')
    model_source=harness.inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    pool=[m for m in ['glm-5.3','qwen3.8-2.4t-a95b'] if m in model_source['active_models']]
    if not pool:raise ValueError('approved_pool_empty')
    names=set(json.loads((ROOT/'work/skills_source_facts_protocol_20261101_r4/source_sha256.json').read_text()))
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py'))
    names.update(['policies/hybrid_skills_planner.py','policies/static_attack_catalog.py','skills/riskweaver/rw-static-fallback/SKILL.md'])
    hashes={name:sha(ROOT/name) for name in sorted(names)}
    protocol=dict(version='hybrid_rag_highlevel_review10_v1',arms=ARMS,scenes=SCENES,planned=20,
        source_scope='original_previously_examined_review_ten_initial_states_not_project_unseen',
        dynamic_catalog_scope='same_previous_frozen_published_dynamic_catalogs; static_catalogs_rebuilt_from_current_state_and_map',
        maximum_physical_requests_per_state_arm=4,maximum_physical_requests=80,
        common_transport=dict(timeout_seconds=180,max_tokens=8192,reasoning_effort='low',SDK_retries=0),
        inputs_sha256=sha(PROTOCOL/'inputs.json'),config_sha256=sha(PROTOCOL/'resolved_config.yaml'),
        inherited_seed_sha256=sha(frozen),inherited_records=len(inherited),seed_source=str(seed),
        frozen_model_pool=pool,model_source=model_source,free_protection_receipt='work/bailian_pool_protection_confirmed_20261023.json',
        static_fallback_enabled=True,source_fact_contract_enabled=True,attack_mode='joint',
        common_instruction=harness.base.INSTRUCTION,
        KB_scope='separate_new_inherited_KB_per_arm; frozen_planning_observations_saved_separately_not_fake_executed_attack_cases',
        target_accepted_per_arm=10,new_execution_frames=0,D=None,majority_correct_not_automatically_certified=True,goal_complete=False)
    harness.base.save(PROTOCOL/'protocol.json',protocol);harness.base.save(PROTOCOL/'source_sha256.json',hashes)
    for name in hashes:
        path=PROTOCOL/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,path)
    print(json.dumps(dict(status='prepared',planned=20,models=pool,frozen_sources=len(hashes),inherited_records=len(inherited),API_requests=0)))


class BoundedPlanner(ORIGINAL_PLANNER):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.client is None:return
        self.client=self.client.with_options(timeout=180,max_retries=0)
        create=self.client.chat.completions.create
        def bounded(*args,**kwargs):
            values=dict(kwargs,max_tokens=8192,reasoning_effort='low')
            values['extra_body']=dict(values.get('extra_body') or {},enable_thinking=True)
            record=dict(model=values['model'],max_tokens=8192,reasoning_effort='low',timeout_seconds=180,
                extra_body=values['extra_body'],messages_sha256=hashlib.sha256(json.dumps(values['messages'],
                    sort_keys=True,ensure_ascii=False).encode()).hexdigest())
            with (CURRENT['output']/'transport_parameters.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(record)+'\n')
            return create(*args,**values)
        self.client.chat.completions.create=bounded


def engine_for(mode,cfg,sample):
    dynamic=ORIGINAL_ENGINE(mode,cfg,sample)
    engine=DynamicFirstStaticPlanner(dynamic,cfg.sim,4)
    runtime=AttackRAGRuntime(cfg.sim,CURRENT['store']);CURRENT['runtime']=runtime
    engine.rag_runtime=runtime;plan=engine.plan
    geometry=pickle.loads(Path(sample['scenario_path']).read_bytes())
    def controlled(planner,env,state,*args,**kwargs):
        env.scenario_dict=geometry;env.cfg=cfg
        env.get_static_obstacles=lambda:copy.deepcopy(state.get('static_obstacles',[]))
        runtime.begin_attempt(env,CURRENT['output'],scenario_source=sample['scenario_path'],metric_source='e1_formal_per_transition')
        runtime.begin_event(env,state,harness.base.INSTRUCTION,.5)
        result=plan(planner,env,state,*args,**kwargs)
        runtime.planned(result,planner.last_trace,planner.last_request_failed)
        observation=copy.deepcopy(runtime.pending)
        observation.update(record_type='planning_observation',status='planned_only_unexecuted',D=None,
            not_a_closed_attack_event=True,actual_attack_execution_frames=0,actual_obstacle_exposure_frames=0)
        harness.base.save(CURRENT['output']/'planning_observation.json',observation)
        return result
    engine.plan=controlled
    return engine


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only')
    protocol=json.loads((PROTOCOL/'protocol.json').read_text());hashes=json.loads((PROTOCOL/'source_sha256.json').read_text())
    if not (ROOT/protocol['free_protection_receipt']).is_file():raise ValueError('free_protection_missing')
    for file,key in [('inputs.json','inputs_sha256'),('resolved_config.yaml','config_sha256'),('inherit_seed.json','inherited_seed_sha256')]:
        if sha(PROTOCOL/file)!=protocol[key]:raise ValueError('protocol_data_changed')
    if not harness.base.verify_sources(protocol,hashes)['verified']:raise ValueError('source_changed')
    samples=json.loads((PROTOCOL/'inputs.json').read_text())
    if [s['scene'] for s in samples]!=SCENES or len(samples)*len(ARMS)!=20:raise ValueError('denominator_changed')
    OUTPUT.mkdir(exist_ok=False);OUTPUT.with_suffix('.pid').write_text(str(os.getpid()))
    for file in ['protocol.json','source_sha256.json','inputs.json','resolved_config.yaml','inherit_seed.json']:
        shutil.copyfile(PROTOCOL/file,OUTPUT/file)
    shutil.copytree(PROTOCOL/'source',OUTPUT/'source_snapshot')
    harness.base.save(OUTPUT/'launch.json',dict(pid=os.getpid(),started_at=time.time(),planned=20,CPU_only=True,new_execution_frames=0))
    configs={};stores={};initial={}
    for arm in ARMS:
        directory=OUTPUT/arm;directory.mkdir();cfg=OmegaConf.load(PROTOCOL/'resolved_config.yaml');cfg.sim.rag.enabled=arm=='skills_rag'
        receipt=initialize_rag_experiment(cfg.sim,directory);store=AttackRAGStore(receipt['knowledge_path'])
        configs[arm]=cfg;stores[arm]=store;initial[arm]=sha(store.path);OmegaConf.save(cfg,directory/'resolved_config.yaml')
    rows=[];checks=[];restore=harness.base.install_control('service_only')
    harness.base.engine_for=engine_for;harness.base.LLMAdversarialPlanner=BoundedPlanner
    harness.AttackRAGRuntime=lambda *a,**k:CURRENT['runtime']
    try:
        for index,sample in enumerate(samples):
            if sha(sample['scenario_path'])!=sample['scenario_sha256']:raise ValueError('scene_geometry_changed')
            for arm in (ARMS if index%2==0 else list(reversed(ARMS))):
                output=OUTPUT/arm/f'scene{sample["scene"]:03d}'
                CURRENT.update(output=output,store=stores[arm],runtime=None)
                check=harness.base.verify_sources(protocol,hashes);checks.append(check);harness.base.save(OUTPUT/'runtime_source_integrity.json',checks)
                if not check['verified']:raise ValueError('source_changed_during_pair')
                row=harness.run_trial(sample,configs[arm],protocol['frozen_model_pool'],output,stores[arm]);plan=row.get('plan') or {}
                row.update(arm=arm,accepted_dynamic_plan=bool(plan.get('attack')),accepted_static_plan=bool(plan.get('obstacle_plan')),
                    accepted_plan=bool(plan.get('attack') or plan.get('obstacle_plan')),
                    scope='frozen_highlevel_review_initial_state_not_full_episode_or_actual_attack',semantic_review='not_reviewed')
                harness.base.save(output/'result.json',row);rows.append(row);harness.base.save(OUTPUT/'results.partial.json',dict(planned=20,results=rows))
                if row.get('runtime_failure') or row.get('planner_service_failure'):raise RuntimeError('final_service_or_runtime_failure')
        if any(sha(stores[arm].path)!=initial[arm] for arm in ARMS):raise ValueError('planning_only_probes_must_not_fabricate_attack_KB_outcomes')
        harness.base.save(OUTPUT/'results.json',dict(status='completed',planned=20,results=rows,
            final_source_check=harness.base.verify_sources(protocol,hashes),KB_seed_copies_unchanged=True,
            new_execution_frames=0,D=None,majority_correct_planning_certified=False,goal_complete=False))
    except Exception as error:
        harness.base.save(OUTPUT/'results.json',dict(status='failed',error_type=type(error).__name__,planned=20,
            results=rows,unattempted=20-len(rows),new_execution_frames=0,D=None,goal_complete=False));raise
    finally:restore()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else run()
