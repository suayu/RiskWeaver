"""复现此前numpy.bool_故障并核验失败写入不损坏已存原件。"""
import json
import numpy as np
import pytest
from json_io import save


def test_nested_numpy_values_roundtrip(tmp_path):
    path = tmp_path / 'scene.json'
    value = {'accepted': np.bool_(True), 'rejected': np.bool_(False),
             'rows': [{'count': np.int64(2), 'margin': np.float64(.5),
                       'positions': np.array([[1., 2.], [3., 4.]])}], 'D': None}
    save(path, value)
    assert json.loads(path.read_text()) == {'accepted': True, 'rejected': False,
        'rows': [{'count': 2, 'margin': .5, 'positions': [[1., 2.], [3., 4.]]}], 'D': None}


def test_unknown_type_fails_before_replacing_previous_result(tmp_path):
    path = tmp_path / 'progress.json'
    path.write_text('{"scanned": 1}')
    before = path.read_bytes()
    with pytest.raises(TypeError):
        save(path, {'bad': object()})
    assert path.read_bytes() == before
    assert not path.with_suffix('.json.pending').exists()
