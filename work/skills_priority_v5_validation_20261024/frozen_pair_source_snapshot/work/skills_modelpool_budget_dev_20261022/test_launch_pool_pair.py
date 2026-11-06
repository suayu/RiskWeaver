"""资源、源码和防覆盖测试；不启动真实实验或模型服务。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
import launch_pool_pair as launcher
from model_pool_protocol import inspect_model_source


def test_existing_cuda_process_blocks_even_idle_gpu():
    gpu='0, GPU-A, 0, 32000\n1, GPU-B, 0, 32000\n2, GPU-C, 90, 25000\n3, GPU-D, 0, 19000\n'
    assert launcher.idle_from_rows(gpu,'GPU-A\n')==['1']
    assert launcher.idle_from_rows(gpu,'GPU-A\n',{'1'})==[]


def test_child_changes_only_output_manifest_and_confirmation():
    original=['python','entry.py','--manifest','old.json','--output-dir','old','--model','a','--models','a','b']
    prepared={'launch_command_pending_confirmation':original}
    actual=launcher.child_command(prepared,Path('/new/manifest.json'),Path('/new/output'))
    assert actual[:3]==['bash','scripts/run_with_llm_provider.sh','dashscope']
    assert actual[actual.index('--manifest')+1]=='/new/manifest.json'
    assert actual[actual.index('--output-dir')+1]=='/new/output'
    assert actual[-1]=='--free-protection-confirmed' and original[3]=='old.json'


def test_source_guard_allows_only_model_comment_not_algorithm_change(tmp_path):
    path=tmp_path/'policies/llm_adversarial_planner.py';path.parent.mkdir()
    path.write_text('class LLMAdversarialPlanner:\n    FREE_MODEL_POOL = (\n        "a",\n        "b",\n    )\n')
    other=tmp_path/'core.py';other.write_text('VALUE=1\n')
    protocol={'model_source':inspect_model_source(path)}
    hashes={'core.py':hashlib.sha256(other.read_bytes()).hexdigest(),
            'policies/llm_adversarial_planner.py':hashlib.sha256(path.read_bytes()).hexdigest()}
    path.write_text(path.read_text().replace('        "a",','        # "a",  # 百炼模型404，自动停用'))
    assert launcher.verify_sources(protocol,hashes,tmp_path)['verified']
    other.write_text('VALUE=2\n')
    assert not launcher.verify_sources(protocol,hashes,tmp_path)['verified']


def test_missing_confirmation_creates_no_output(tmp_path,monkeypatch):
    output=tmp_path/'no_output'
    monkeypatch.setattr(sys,'argv',['entry','--protocol',str(tmp_path/'missing'),'--stage','development','--output-dir',str(output)])
    with pytest.raises(ValueError,match='free_only_protection_confirmation_required'):launcher.main()
    assert not output.exists()


def test_failed_launch_cannot_overwrite_historical_result(tmp_path):
    output=tmp_path/'historical';output.mkdir();marker=output/'launcher_result.json'
    marker.write_text('{"status":"historical-do-not-overwrite"}')
    before=marker.read_bytes()
    result=subprocess.run([sys.executable,launcher.__file__,'--protocol',str(tmp_path/'missing'),
        '--stage','development','--output-dir',str(output),'--free-protection-confirmed'],capture_output=True,text=True)
    assert result.returncode and marker.read_bytes()==before
