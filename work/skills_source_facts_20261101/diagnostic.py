"""四个已知事实反例的真实高层诊断；冻结旧输入，不执行仿真或产生虚构攻击事件。"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
import uuid

ROOT = Path('/home2/zhaoyx/scenario-dreamer'); HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'work/skills_highlevel_comparison_20261030'),
               str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
import numpy as np
from omegaconf import OmegaConf
import frozen_planning as base
from model_pool_protocol import inspect_model_source
from policies.rag_runtime import AttackRAGRuntime, initialize_rag_experiment
from policies.rag_store import AttackRAGStore
from policies.skills_source_facts import source_facts

OLD = ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2'
PROTOCOL = ROOT/'work/skills_source_facts_protocol_20261101_r1'
OUTPUT = ROOT/'experiments/skills_source_facts_frozen4_diagnostic_20261101_r1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    PROTOCOL.mkdir(exist_ok=False)
    samples = []
    for scene, step in [(74, 66), (182, 22), (182, 33)]:
        path = OLD/f'skills_no_rag/movies/scenario_{scene:03d}/execution_trace.jsonl'
        rows = [json.loads(line) for line in path.open(encoding='utf-8')]
        frame = next(row for row in rows if row['kind'] == 'state' and row['step'] == step)
        state = next(row['state'] for row in rows if row['kind'] == 'llm_input' and row['step'] == step)
        trace = next(row['planner_trace'] for row in rows if row['kind'] == 'llm_output' and row['step'] == step)
        context = trace['context']
        samples.append(dict(scene=scene, scene_id=context['scene_id'], source_step=step,
            state=state, frame={k:frame[k] for k in ['ego', 'agents', 'active']},
            contexts=dict(temporal_original=context), source_trace=str(path), source_trace_sha256=sha(path),
            previous_response=trace['response_summary'], scope='known_development_counterexample'))
    path = ROOT/'work/skills_candidate_local_holdout_protocol_20261030_r1/inputs.json'
    sample = next(s for s in json.loads(path.read_text()) if s['scene'] == 146)
    samples.append(dict(sample, scope='previously_reviewed_lateral_deadband_counterexample'))
    cfg = OmegaConf.load(OLD/'skills_no_rag/resolved_config.yaml')
    cfg.sim.llm.skills.source_fact_contract_enabled = True
    cfg.sim.llm.skills.grounding_validation_mode = 'candidate_local'
    cfg.sim.rag.enabled = False; cfg.sim.rag.record_events = True
    cfg.sim.rag.knowledge_mode = 'new'; cfg.sim.rag.inherit = False; cfg.sim.rag.inherit_path = None
    cfg.sim.rag.runtime_knowledge_path = None; cfg.sim.rag.runtime_experiment_id = None
    model_source = inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    preferred = ['qwen3.8-max','qwen3.7-flash','glm-5.3','qwen3.8-2.4t-a95b','deepseek-v4.1-flash-1017']
    pool = [name for name in preferred if name in model_source['active_models']]
    if not pool:raise ValueError('approved_model_pool_empty')
    # 真实装配但零API预检：验证冻结的后续帧没有被悄悄重设为第0帧。
    readiness = []
    for sample in samples:
        engine = base.engine_for('temporal_skills_common', cfg, sample)
        facts = source_facts(sample['state'], sample['contexts']['temporal_original'],
                             sample['source_step'], float(cfg.sim.dt), sample['scene_id'])
        if not facts['available'] or engine.source_fact_contract is None:
            raise ValueError('source_fact_actual_engine_not_ready')
        readiness.append(dict(scene=sample['scene'], step=sample['source_step'], facts=facts,
                              published_candidates=len(sample['contexts']['temporal_original']['candidates'])))
    base.save(PROTOCOL/'inputs.json', samples); OmegaConf.save(cfg, PROTOCOL/'resolved_config.yaml')
    old = ROOT/'work/skills_rag_two_arm_preparation_20261030_r3'
    names = set(json.loads((old/'source_sha256.json').read_text()))
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*.py'))
    names.update(p.relative_to(HERE/'source_after').as_posix() for p in (HERE/'source_after').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    hashes = {name:sha(ROOT/name) for name in sorted(names)}
    protocol = dict(version='source_facts_frozen4_diagnostic_v1', planned=4,
        scene_step_pairs=[[s['scene'],s['source_step']] for s in samples],
        inputs_sha256=sha(PROTOCOL/'inputs.json'), config_sha256=sha(PROTOCOL/'resolved_config.yaml'),
        model_source=model_source, frozen_model_pool=pool, maximum_physical_SDK_per_state=4,
        maximum_physical_SDK=16, free_protection_receipt='work/bailian_pool_protection_confirmed_20261023.json',
        scope='known_counterexample_development_diagnostic_not_generalization_or_closed_loop',
        new_execution_frames=0, D=None, RAG_retrieval=False, append_attack_events=False, goal_complete=False)
    base.save(PROTOCOL/'protocol.json', protocol); base.save(PROTOCOL/'source_sha256.json', hashes)
    base.save(PROTOCOL/'readiness.json', readiness)
    for name in hashes:
        path=PROTOCOL/'source'/name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/name,path)
    print(json.dumps(dict(status='prepared', inputs=4, frozen_sources=len(hashes), API_requests=0)))


def run_trial(sample, cfg, pool, output, store):
    output.mkdir(exist_ok=False)
    engine = base.engine_for('temporal_skills_common', cfg, sample)
    planner = base.LLMAdversarialPlanner(model_name=pool[0], model_names=pool, provider='dashscope')
    if planner.provider != 'dashscope' or not planner.available:raise RuntimeError('Bailian_unavailable')
    planner.use_obstacles=False; planner.allow_trajectory=True; planner.use_multimodal=False
    create=planner.client.chat.completions.create
    def recorded(*args, **kwargs):
        start=time.monotonic(); row=dict(physical_request_id=uuid.uuid4().hex,
            model_requested=kwargs.get('model'), messages=kwargs.get('messages'),
            extra_body=kwargs.get('extra_body'), started_at=time.time())
        try:
            response=create(*args,**kwargs)
            row.update(model_returned=response.model, content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else None,
                finish_reason=response.choices[0].finish_reason)
            return response
        except Exception as error:
            row.update(error_type=type(error).__name__, status_code=getattr(error,'status_code',None)); raise
        finally:
            row['elapsed_seconds']=time.monotonic()-start
            with (output/'api_calls.jsonl').open('a',encoding='utf-8') as file:
                file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    planner.client.chat.completions.create=recorded
    budget=base.PhysicalCallBudget(4,output/'budget_events.jsonl'); budget.reset(sample['scene']); budget.attach(planner)
    original=base.LLMAdversarialPlanner._request_attack_plan
    planner._request_attack_plan=lambda prompt,scene_image=None:base.bounded_model_request(planner,original,prompt,scene_image=scene_image)
    state=copy.deepcopy(sample['state']); frame=sample['frame']
    env=NS(current_step=sample['source_step'], current_scene_id=sample['scene_id'], dt=float(cfg.sim.dt),
        agent_active=np.asarray(frame['active']), ego_state=np.asarray(frame['ego']),
        data_dict={'agent':[np.asarray(frame['agents'])]}, get_state_for_planning=lambda:state)
    # 仅复用无检索的固定规则桥，与上一轮真实请求相同；没有伪造执行/失败事件写入KB。
    runtime=AttackRAGRuntime(cfg.sim,store); started=time.monotonic()
    result=dict(scene=sample['scene'],source_step=sample['source_step'],planned=True,attempted=True,
        accepted_plan=False,new_execution_frames=0,D=None,semantic_review='not_reviewed',
        scope='known_counterexample_frozen_highlevel_only')
    try:
        with runtime.prompt_bridge(planner):
            plan=engine.plan(planner,env,state,base.INSTRUCTION)
        result.update(plan=plan,accepted_plan=bool(plan and plan.get('attack') is True),
            planner_service_failure=planner.last_request_failed,planner_trace=planner.last_trace)
    except Exception as error:
        result.update(runtime_failure=True,error_type=type(error).__name__)
    finally:
        result.update(budget=budget.snapshot(),elapsed_seconds=time.monotonic()-started)
        base.save(output/'result.json',result)
    return result


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only_required')
    protocol=json.loads((PROTOCOL/'protocol.json').read_text()); hashes=json.loads((PROTOCOL/'source_sha256.json').read_text())
    if not (ROOT/protocol['free_protection_receipt']).is_file():raise ValueError('protection_receipt_missing')
    if not base.verify_sources(protocol,hashes)['verified']:raise ValueError('source_changed')
    if sha(PROTOCOL/'inputs.json')!=protocol['inputs_sha256'] or sha(PROTOCOL/'resolved_config.yaml')!=protocol['config_sha256']:
        raise ValueError('input_or_config_changed')
    samples=json.loads((PROTOCOL/'inputs.json').read_text())
    if len(samples)!=4 or [[s['scene'],s['source_step']] for s in samples]!=protocol['scene_step_pairs']:
        raise ValueError('denominator_changed')
    OUTPUT.mkdir(exist_ok=False); OUTPUT.with_suffix('.pid').write_text(str(os.getpid()))
    for name in ['protocol.json','inputs.json','source_sha256.json','resolved_config.yaml','readiness.json']:
        shutil.copyfile(PROTOCOL/name,OUTPUT/name)
    shutil.copytree(PROTOCOL/'source',OUTPUT/'source_snapshot')
    cfg=OmegaConf.load(PROTOCOL/'resolved_config.yaml')
    receipt=initialize_rag_experiment(cfg.sim,OUTPUT); store=AttackRAGStore(receipt['knowledge_path']); before=sha(store.path)
    base.save(OUTPUT/'launch.json',dict(pid=os.getpid(),started_at=time.time(),planned=4,CPU_only=True,new_execution_frames=0))
    rows=[]; checks=[]; restore=base.install_control('service_only')
    try:
        for sample in samples:
            check=base.verify_sources(protocol,hashes); checks.append(check)
            base.save(OUTPUT/'runtime_source_integrity.json',checks)
            if not check['verified']:raise ValueError('source_changed_during_diagnostic')
            row=run_trial(sample,cfg,protocol['frozen_model_pool'],OUTPUT/f'scene{sample["scene"]:03d}_step{sample["source_step"]:03d}',store)
            rows.append(row);base.save(OUTPUT/'results.partial.json',dict(planned=4,results=rows))
            if row.get('runtime_failure') or row.get('planner_service_failure'):raise RuntimeError('final_service_or_runtime_failure')
        if sha(store.path)!=before:raise ValueError('planning_only_probe_must_not_fabricate_KB_outcomes')
        base.save(OUTPUT/'results.json',dict(status='completed',planned=4,results=rows,
            final_source_check=base.verify_sources(protocol,hashes),KB_unchanged=True,
            new_execution_frames=0,D=None,majority_correct_planning_certified=False,goal_complete=False))
    except Exception as error:
        base.save(OUTPUT/'results.json',dict(status='failed',error_type=type(error).__name__,planned=4,
            results=rows,unattempted=4-len(rows),new_execution_frames=0,D=None,goal_complete=False));raise
    finally:restore()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();prepare() if args.prepare else run()
