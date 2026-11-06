"""控制模式、严格规划契约和服务失败的轻量回归。"""
import json
from pathlib import Path
import threading
import time
import uuid

import pytest

from policies.codex_planner_bridge import CodexQueueClient, PlannerServiceFailure, validate_plan
from policies.difficulty_control import DifficultyController
from policies.planner_smoke_gate import validate_planner_smoke


def plan(rid):
    return dict(request_id=rid, target_id=1, strategy='slow_down', anchors=[[0, 12], [0, 20], [0, 27], [0, 33]], duration=10, no_attack_reason=None)


@pytest.mark.parametrize('observed', [.1, .8, None, float('nan')])
def test_target_ignores_collision(observed):
    controller = DifficultyController({'mode': 'target'})
    yes = controller.update(.5, observed, True, .5, .1)
    no = controller.update(.5, observed, False, .5, .1)
    assert yes['after'] == no['after']
    assert yes['reason'] == no['reason']


def test_adaptive_collision_and_off():
    assert DifficultyController({'mode': 'adaptive'}).update(.5, .1, True, .5, .1)['after'] == .35
    assert DifficultyController({'mode': 'off'}).update(.5, .1, True, .5, .1)['after'] == .5


def test_weights_change_optimizer_fixed():
    controller = DifficultyController({'mode': 'target'})
    parameters = [controller.parameters(u / 100) for u in range(101)]
    assert len({p['llm_anchor'] for p in parameters}) == 101
    assert len({p['scenario_ttc'] for p in parameters}) == 101
    assert len({(p['inner_lr'], p['inner_beta'], p['n_guide_steps']) for p in parameters}) == 1


@pytest.mark.parametrize('value', [-1, 9, 2.5, float('nan')])
def test_invalid_fixed_optimizer(value):
    with pytest.raises(ValueError):
        DifficultyController({'fixed_optimizer': dict(inner_lr=.14, inner_beta=.35, n_guide_steps=value)})


@pytest.mark.parametrize('field,value', [('trajectory', []), ('request_id', 'bad'), ('duration', 0), ('anchors', [[0, 0]] * 30), ('target_id', True)])
def test_schema_rejects_wrong_id_dense_trajectory_and_invalid_fields(field, value):
    rid = uuid.uuid4().hex
    payload = plan(rid)
    payload[field] = value
    with pytest.raises(PlannerServiceFailure):
        validate_plan(payload, rid)


def test_queue_response_and_trace(tmp_path):
    client = CodexQueueClient(tmp_path / 'queue', timeout=2)
    def worker():
        for _ in range(100):
            pending = list(client.directory.glob('*/pending.json'))
            if pending:
                rid = pending[0].parent.name
                response = dict(request_id=rid, status='ok', exit_code=0, plan=plan(rid))
                temp = pending[0].parent / 'result.tmp'
                temp.write_text(json.dumps(response))
                temp.rename(temp.with_name('response.json'))
                return
            time.sleep(.01)
    thread = threading.Thread(target=worker)
    thread.start()
    assert client.request({'state': {}})['target_id'] == 1
    thread.join()
    assert client.last_trace['validation'] == 'schema_valid'


def test_timeout_is_service_failure(tmp_path):
    client = CodexQueueClient(tmp_path / 'queue', timeout=1)
    with pytest.raises(PlannerServiceFailure, match='timeout'):
        client.request({})
    assert client.last_trace['validation'] == 'planner_service_failure'


def test_smoke_fails_closed_without_attack_or_on_service_failure():
    with pytest.raises(RuntimeError, match='planner_service_failure'):
        validate_planner_smoke({'planner_service_failure_count': 1})
    with pytest.raises(RuntimeError, match='no_attack_execution'):
        validate_planner_smoke({})
