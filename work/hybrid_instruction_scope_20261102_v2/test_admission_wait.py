"""原资源入口的零GPU/零API执行回归，覆盖显存变化与方法恢复。"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from admission_wait import wait_before_loading

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
CAPPED = ROOT / 'work/skills_shared_resource_20261028/run_capped.py'


def test_actual_original_resource_main_waits_then_keeps_cap(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CAPPED.parent))
    spec = importlib.util.spec_from_file_location('actual_cap_zero_GPU_fixture', CAPPED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in CAPPED.parent.glob('*.py')}
    (tmp_path / 'source_sha256.json').write_text(json.dumps(files))
    free_values = iter([16 * 2**30, 17 * 2**30])
    fractions, waits = [], []
    def memory(*args):
        return next(free_values, 18 * 2**30), 32 * 2**30
    cuda = NS(mem_get_info=memory, is_available=lambda: True, device_count=lambda: 1,
              get_device_properties=lambda _: NS(total_memory=32 * 2**30),
              set_per_process_memory_fraction=lambda fraction, **kw: fractions.append(fraction),
              reset_peak_memory_stats=lambda _: None,
              memory_allocated=lambda _: 0, memory_reserved=lambda _: 0,
              max_memory_allocated=lambda _: 0, max_memory_reserved=lambda _: 0)
    out = tmp_path / 'arm'
    def no_model_run():
        out.mkdir()
        return 0
    monkeypatch.setitem(sys.modules, 'torch', NS(cuda=cuda))
    monkeypatch.setitem(sys.modules, 'run_controlled', NS(main=no_model_run))
    monkeypatch.setattr(sys, 'argv', ['fixture', '--resource-cap-gib', '14',
        '--resource-reserve-gib', '3', '--output-dir', str(out)])
    status = tmp_path / 'admission.json'
    with wait_before_loading(cuda, 17 * 2**30, status, CAPPED, ROOT, wait=waits.append):
        assert module.main() == 0
    assert waits == [15]
    assert fractions == [14 / 32]
    assert cuda.mem_get_info is memory
    assert json.loads(status.read_text())['status'] == 'admitted_before_model_load'
    receipt = json.loads((tmp_path / 'arm_resource_result.json').read_text())
    assert receipt['cap_gib'] == 14 and receipt['reserve_gib'] == 3
    assert receipt['resource_fixture'] == 'off'
    assert receipt['sampling_synchronization_for_timing'] is False


def test_unrelated_memory_query_is_not_delayed(tmp_path):
    original = lambda *args: (1, 32 * 2**30)
    cuda = NS(mem_get_info=original)
    with wait_before_loading(cuda, 17 * 2**30, tmp_path / 'status.json', CAPPED, ROOT,
                             wait=lambda _: pytest.fail('unrelated call waited')):
        assert cuda.mem_get_info(0)[0] == 1
    assert cuda.mem_get_info is original
    assert not (tmp_path / 'status.json').exists()


def test_interface_restored_if_worker_raises(tmp_path):
    original = lambda *args: (18 * 2**30, 32 * 2**30)
    cuda = NS(mem_get_info=original)
    with pytest.raises(RuntimeError):
        with wait_before_loading(cuda, 17 * 2**30, tmp_path / 'status.json', CAPPED, ROOT):
            raise RuntimeError('fixture')
    assert cuda.mem_get_info is original
