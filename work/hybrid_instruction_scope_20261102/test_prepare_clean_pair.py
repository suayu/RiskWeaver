"""新协议的范围与参数不变性回归，无请求、仿真或写入正式协议。"""
import copy
import json
from pathlib import Path
import pytest
from prepare_clean_pair import retarget, BASE


def test_retarget_preserves_all_algorithm_flags_and_original():
    old = json.loads((BASE / 'protocol.json').read_text())
    before = copy.deepcopy(old)
    target, output = Path('/tmp/protocol'), Path('/tmp/result')
    new = retarget(old, target, output)
    assert old == before
    for arm in old['arms']:
        previous = old['commands'][arm]
        actual = new['commands'][arm]
        assert len(actual) == len(previous)
        allowed = {1, previous.index('--manifest') + 1, previous.index('--output-dir') + 1}
        assert all(a == b for i, (a, b) in enumerate(zip(previous, actual)) if i not in allowed)
        assert actual[actual.index('--rag') + 1] == previous[previous.index('--rag') + 1]
    for key in ['models', 'planned_scenes', 'maximum_frames', 'maximum_physical_SDK_per_scene',
                'worker_cap_gib', 'worker_reserve_gib', 'manifest_sha256', 'cold_KB_per_arm']:
        assert new[key] == old[key]


@pytest.mark.parametrize('key,value', [('planned_attempts', 20), ('models', ['other']),
                                     ('maximum_physical_SDK_per_scene', 5)])
def test_retarget_rejects_scope_or_budget_drift(key, value):
    old = json.loads((BASE / 'protocol.json').read_text())
    old[key] = value
    with pytest.raises(ValueError):
        retarget(old, Path('/tmp/p'), Path('/tmp/e'))
