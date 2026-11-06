"""真实冻结状态、RAG和动态/静态工厂共同装配；仅替代SDK，不作性能证据。"""
from pathlib import Path
import copy,json,pickle,sys
from types import SimpleNamespace as NS
import numpy as np
import pytest
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(Path(__file__).resolve().parent),str(ROOT/'work/skills_scene_facts_prototype_20261028')]
import run_pair as subject
from test_fact_runtime import answer
from policies.rag_store import AttackRAGStore
from policies.rag_runtime import initialize_rag_experiment


@pytest.mark.parametrize('scene',[6,28,143,146])
@pytest.mark.parametrize('rag',[False,True])
def test_actual_frozen_two_branch_RAG_wire_keeps_planning_out_of_attack_KB(tmp_path,scene,rag):
    prior=ROOT/'work/skills_candidate_local_holdout_protocol_20261030_r1'
    sample=next(s for s in json.loads((prior/'inputs.json').read_text()) if s['scene']==scene)
    manifest=json.loads((prior/'validation10.json').read_text());index=manifest['original_scenario_indices'].index(scene)
    sample['scenario_path']=manifest['scenario_files'][index]
    cfg=OmegaConf.load(ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
    cfg.sim.llm.skills.source_fact_contract_enabled=True;cfg.sim.llm.skills.static_fallback_enabled=True
    cfg.sim.llm.skills.grounding_validation_mode='candidate_local';cfg.sim.llm.attack_mode='joint';cfg.sim.llm.obstacles.enabled=False
    cfg.sim.rag.enabled=rag;cfg.sim.rag.knowledge_mode='new';cfg.sim.rag.inherit=True
    cfg.sim.rag.inherit_path=str(next((ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_rag/rag_knowledge').glob('*.json')))
    receipt=initialize_rag_experiment(cfg.sim,tmp_path)
    store=AttackRAGStore(receipt['knowledge_path']);before=store.path.read_bytes()
    output=tmp_path/'planning_only';output.mkdir()
    subject.CURRENT.update(output=output,store=store,runtime=None)
    engine=subject.engine_for('temporal_skills_common',cfg,sample);runtime=subject.CURRENT['runtime']
    frame=sample['frame'];state=copy.deepcopy(sample['state'])
    env=NS(current_step=sample['source_step'],current_scene_id=sample['scene_id'],dt=.1,
        ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
        agent_active=np.asarray(frame['active']),get_state_for_planning=lambda:state)
    # 时序AST守卫要求真实生产方法的源码，不能使用AST重建的旧测试替身。
    planner=subject.ORIGINAL_PLANNER(model_name='glm-5.3',model_names=['glm-5.3'],provider='dashscope',client=NS())
    planner.use_obstacles=True;planner.allow_trajectory=True;calls=[]
    def request(prompt,scene_image=None):
        payload=json.loads(prompt.rsplit('\n',1)[1]);calls.append(payload)
        if payload['context']['candidates']:response=answer(payload)
        else:
            response={k:payload[k] for k in ['request_id','state_hash','catalog_hash']}
            response.update(ranking=[],scene_analysis='No usable vehicle; choose route obstruction.',
                candidate_reasons={},no_attack_reason='Static attack instead of dynamic attack.',
                obstacle_candidate_id=payload['context']['obstacle_candidates'][0]['obstacle_candidate_id'])
        return '',response
    planner._request_attack_plan=request
    with runtime.prompt_bridge(planner):plan=engine.plan(planner,env,state,subject.harness.base.INSTRUCTION)
    assert plan and (plan.get('attack') or plan.get('obstacle_plan'))
    assert len(calls)==1 and ('retrieved_attack_evidence' in calls[0]['context']) is rag
    assert calls[0]['context']['source_facts_v1']['available']
    observation=json.loads((output/'planning_observation.json').read_text())
    assert observation['not_a_closed_attack_event'] and observation['D'] is None
    assert store.path.read_bytes()==before
