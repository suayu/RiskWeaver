"""真实时序/身份/候选校验/RAG装配的请求端测试；SDK使用替身，不发真实请求。"""
from pathlib import Path
import copy
import json
import sys

import pytest
from omegaconf import OmegaConf

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0] = [str(ROOT), str(ROOT/'work/skills_temporal_candidates_20261029'),
               str(ROOT/'work/skills_identity_evidence_20261030'),
               str(ROOT/'work/skills_scene_facts_prototype_20261028')]
from temporal_planner import TemporalCandidatePlanner
from replay_074 import source_fixture
from common_evidence import install_common_evidence
from test_fact_runtime import answer
from policies.rag_runtime import AttackRAGRuntime
from policies.skills_source_facts import FIELD, RULE


def execute(enabled, rag=None, corrupt_first=False):
    env, state, planner, *_ = source_fixture()
    cfg = OmegaConf.create(OmegaConf.to_container(env.cfg, resolve=True))
    cfg.sim.llm.skills.grounding_validation_mode = 'candidate_local'
    cfg.sim.llm.skills.source_fact_contract_enabled = enabled
    engine = TemporalCandidatePlanner(cfg.sim, 4, extra_recipes=True, rolling_intent=True,
        explicit_contract=True, deduplicate=True, fractional_interaction=True,
        velocity_continuation=True, motion_evidence=True)
    engine = install_common_evidence(engine)
    if rag is not None:
        # 仅检验实际桥接器的装配顺序；虚构案例不是检索收益或真实 API 证据。
        runtime = AttackRAGRuntime.__new__(AttackRAGRuntime)
        runtime.enabled = rag
        runtime.current_evidence = dict(documents=[dict(case_id='wiring-fixture')]) if rag else None
        runtime.rules_documents = lambda: []
        runtime.prepare_context = lambda state, context, env: context
        runtime.bind_planner(engine)
    calls = []; outputs = []
    def request(prompt, scene_image=None):
        payload = json.loads(prompt.rsplit('\n', 1)[1]); calls.append(payload)
        response = answer(payload)
        if corrupt_first and len(calls) == 1:
            response['state_hash'] = 'intentionally_invalid_identity'
        outputs.append(copy.deepcopy(response)); return '', response
    planner._request_attack_plan = request
    plan = engine.plan(planner, env, state, ['probe'])
    return plan, planner, calls, outputs


@pytest.mark.parametrize('rag', [None, False, True])
def test_real_nested_prompt_has_facts_and_same_original_validation(rag):
    plan, planner, calls, outputs = execute(True, rag)
    assert plan and plan['attack'] and len(calls) == 1
    payload = calls[0]
    assert payload['context'][FIELD]['available']
    assert any(doc['path'] == RULE for doc in payload['knowledge']['documents'])
    assert planner.last_trace['response_summary'] == outputs[0]
    assert planner.last_trace['plan_validation'] == 'accepted'
    assert len(planner.last_trace['source_fact_contract']['actual_wire']) == 1
    assert ('retrieved_attack_evidence' in payload['context']) == (rag is True)
    assert payload['required_identity_copy_exactly']['state_hash'] == payload['state_hash']


def test_disabled_keeps_old_wire_and_candidate_order():
    old_plan, old, old_calls, _ = execute(False)
    new_plan, new, new_calls, _ = execute(True)
    assert FIELD not in old_calls[0]['context']
    assert not any(doc['path'] == RULE for doc in old_calls[0]['knowledge']['documents'])
    assert 'source_fact_contract' not in old.last_trace
    assert old_calls[0]['context']['candidates'] == new_calls[0]['context']['candidates']
    assert old_plan['candidate_id'] == new_plan['candidate_id']
    assert old_plan['anchors'] == new_plan['anchors']


def test_real_repair_keeps_bad_response_and_facts_without_extra_call():
    plan, planner, calls, outputs = execute(True, True, corrupt_first=True)
    assert plan and plan['attack'] and len(calls) == 2
    assert planner.last_trace['requests'][0]['response']['state_hash'] == 'intentionally_invalid_identity'
    assert planner.last_trace['requests'][1]['response'] == outputs[1]
    assert calls[0]['context'][FIELD] == calls[1]['context'][FIELD]
    assert planner.last_trace['api_calls_total'] == 2
    assert planner.last_trace['schema_rejections'][0]['reason'] == 'skills_identity_mismatch'
