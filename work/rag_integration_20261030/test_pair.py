"""两组开关/配置/继承隔离与已批准模型过滤的离线检查。"""
from pathlib import Path
import importlib.util,json,sys
import pytest
from hydra import compose,initialize_config_dir
from policies.rag_runtime import initialize_rag_experiment
from policies.rag_store import AttackRAGStore

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rag_entry',HERE/'run_rag_worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


@pytest.mark.parametrize('inherit',[False,True])
def test_hydra_two_arms_same_rules_and_separate_new_KB(tmp_path,inherit):
    seed=None
    if inherit:
        old=AttackRAGStore.create_new(tmp_path/'old');seed=old.path;before=seed.read_bytes()
        old.append(dict(case_id='past',record_type='attack_event',status='closed',D=None,measurement_valid=False,policy_id='old'))
        before=seed.read_bytes()
    values=[];receipts=[]
    for enabled in (False,True):
        with initialize_config_dir(version_base=None,config_dir=str(ROOT/'cfgs')):
            cfg=compose(config_name='config',overrides=worker.overrides(enabled,seed)+['sim.llm.skills.enabled=true'])
        values.append(cfg)
        receipts.append(initialize_rag_experiment(cfg.sim,tmp_path/str(enabled)))
    assert values[0].sim.rag.enabled is False and values[1].sim.rag.enabled is True
    assert receipts[0]['knowledge_path']!=receipts[1]['knowledge_path']
    assert receipts[0]['initial_records']==receipts[1]['initial_records']==int(inherit)
    if inherit:
        assert seed.read_bytes()==before
        assert receipts[0]['lineage']['lineage']['source_sha256']==receipts[1]['lineage']['lineage']['source_sha256']
    first=AttackRAGStore(receipts[0]['knowledge_path']);second=AttackRAGStore(receipts[1]['knowledge_path'])
    first.append(dict(case_id='first_only',record_type='attack_event',status='closed',D=None,measurement_valid=False))
    assert not any(r['case_id']=='first_only' for r in second.snapshot()['records'])


def test_shared_GPU_startup_headroom_does_not_lower_runtime_reserve():
    sys.path[:0]=[str(ROOT/'work/skills_modelpool_budget_dev_20261022'),str(ROOT/'work/skills_candidate_local_runtime_20261030')]
    spec=importlib.util.spec_from_file_location('rag_queue',HERE/'run_pair.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.select_rag_gpu('3, Tesla V100S-PCIE-32GB, 17545, 100') is None
    assert module.select_rag_gpu('3, Tesla V100S-PCIE-32GB, 18432, 100')=='3'


def test_RAG_profile_does_not_implicitly_load_legacy_history(monkeypatch):
    from types import SimpleNamespace as NS
    import policies.ego_profile as profile
    captured=[]
    monkeypatch.setattr(profile,'EgoProfile',lambda *_:None)
    monkeypatch.setattr(profile,'CandidateBuilder',lambda *_:None)
    monkeypatch.setattr(profile,'JointTrajectorySafetyGate',lambda *_:NS(dynamic_limits={}))
    monkeypatch.setattr(profile,'AttackKnowledge',lambda policy,paths:captured.append(paths))
    cfg=NS(policy='idm',ego_profile=NS(history_paths=['must_not_implicitly_load.jsonl']),llm=NS(skills=NS(enabled=True)),rag=NS(runtime_knowledge_path='new.json'))
    profile.ProfileAttackPipeline(cfg)
    assert captured[-1]==[]
    cfg.llm.skills.enabled=False;profile.ProfileAttackPipeline(cfg)
    assert captured[-1]==['must_not_implicitly_load.jsonl']
