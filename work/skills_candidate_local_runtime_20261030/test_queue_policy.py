"""已批准共享条件边界：17GiB、单卡、型号和允许列表；不发命令/不占GPU。"""
import pytest
from queue_smoke import select_shared_gpu,worker_environment


@pytest.mark.parametrize('free,expected',[(17407,None),(17408,'1')])
def test_exact_memory_boundary_even_when_busy(free,expected):
    assert select_shared_gpu(f'1,Tesla V100S-PCIE-32GB,{free},100',['1'])==expected


def test_selects_only_one_approved_device_with_largest_headroom():
    rows='0,Tesla V100S-PCIE-32GB,7000,100\n1,Tesla V100S-PCIE-32GB,24000,97\n3,Tesla V100S-PCIE-32GB,25000,97'
    assert select_shared_gpu(rows,['0','1','3'])=='3'
    assert select_shared_gpu(rows,['0','1'])=='1'


def test_wrong_hardware_is_not_inferred_safe_from_free_memory():
    assert select_shared_gpu('2,Other GPU,40000,0',['2']) is None


def test_worker_model_environment_overrides_stale_defaults_without_extra_calls(monkeypatch):
    monkeypatch.setenv('LLM_MODEL_NAME','stale-default');monkeypatch.setenv('LLM_MODEL_NAMES','stale-default')
    env=worker_environment({'frozen_model_pool':['deepseek-v4-pro-0813','glm-5.3']},'3')
    assert env['LLM_MODEL_NAME']=='deepseek-v4-pro-0813' and env['LLM_MODEL_NAMES']=='deepseek-v4-pro-0813,glm-5.3'
    assert env['CUDA_VISIBLE_DEVICES']=='3'
