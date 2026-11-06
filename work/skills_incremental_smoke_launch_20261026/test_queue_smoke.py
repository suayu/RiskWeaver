"""用隔离目录验证派发与历史保护；不请求模型、不分配GPU。"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import pytest
import queue_smoke as q


def fixture(tmp_path, monkeypatch):
    root=tmp_path;protocol=root/q.PROTOCOL;protocol.mkdir(parents=True)
    source=root/'policies/llm_adversarial_planner.py';source.parent.mkdir();source.write_text('# model stub')
    code=root/'algorithm.py';code.write_text('frozen')
    scene=root/'scene.pkl';scene.write_bytes(b'frozen scene')
    helper=root/'work/helper/queue_smoke.py';helper.parent.mkdir(parents=True);helper.write_text('queue fixture')
    monkeypatch.setattr(q,'__file__',str(helper))
    (root/'experiments').mkdir();(root/'confirmation.json').write_text('{}')
    q.save(protocol/'protocol.json',dict(model_source={},free_protection_receipt='confirmation.json',frozen_model_pool=['kimi-k3']))
    q.save(protocol/'source_sha256.json',{'algorithm.py':hashlib.sha256(code.read_bytes()).hexdigest()})
    q.save(protocol/'smoke074.json',dict(original_scenario_indices=[74],scenario_files=[str(scene)],scenario_sha256=[hashlib.sha256(scene.read_bytes()).hexdigest()]))
    command=['python','run.py','--output-dir',str(root/q.OUTPUT/'incremental_full')]
    q.save(protocol/'prepared_smoke.json',dict(command=command))
    seen=[]
    def prepare(command,**kwargs):
        assert '--prepare-only' in command and '--free-protection-confirmed' in command
        return SimpleNamespace(stdout=json.dumps(dict(free_protection_operator_confirmed=True)))
    def popen(command,**kwargs):
        seen.append((command,kwargs))
        return SimpleNamespace(pid=12345,wait=lambda:0)
    deps=dict(idle=lambda reserved:['2'],inspect=lambda path:{},transition=lambda a,b:dict(verified=True),
        prerequisite=lambda *args:dict(passed=True),prepare=prepare,popen=popen)
    return root,deps,seen


def test_existing_output_is_untouched(tmp_path,monkeypatch):
    root,deps,seen=fixture(tmp_path,monkeypatch);out=root/q.OUTPUT;out.mkdir();(out/'sentinel').write_bytes(b'original')
    before={p.name:p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError):q.run(root,**deps)
    assert before=={p.name:p.read_bytes() for p in out.iterdir()} and not seen


def test_drift_stops_before_creation(tmp_path,monkeypatch):
    root,deps,seen=fixture(tmp_path,monkeypatch);(root/'algorithm.py').write_text('changed')
    with pytest.raises(RuntimeError,match='source_drift'):q.run(root,**deps)
    assert not (root/q.OUTPUT).exists() and not seen


def test_busy_gpu_never_dispatches(tmp_path,monkeypatch):
    root,deps,seen=fixture(tmp_path,monkeypatch);deps['idle']=lambda reserved:[]
    # BaseException模拟结束观察，不能被误当作任务终态失败。
    class ObservationEnded(BaseException):pass
    deps['sleep']=lambda seconds:(_ for _ in ()).throw(ObservationEnded())
    with pytest.raises(ObservationEnded):q.run(root,**deps)
    out=root/q.OUTPUT
    assert q.load(out/'queue_status.json')['state']=='waiting_for_completely_idle_gpu'
    assert not seen and not (out/'method_pid.json').exists() and not (out/'launcher_result.json').exists()


def test_idle_handoff_preserves_protocol_and_budget(tmp_path,monkeypatch):
    root,deps,seen=fixture(tmp_path,monkeypatch)
    assert q.run(root,**deps)==0
    assert len(seen)==1 and seen[0][0][:3]==['bash','scripts/run_with_llm_provider.sh','dashscope']
    assert seen[0][1]['env']['CUDA_VISIBLE_DEVICES']=='2'
    out=root/q.OUTPUT
    assert q.load(out/'launch.json')['maximum_physical_requests']==4
    assert q.load(out/'launcher_result.json')['automatic_next_stage'] is False
    assert q.load(out/'runtime_source_integrity.json')['verified']


def test_missing_prerequisite_never_starts(tmp_path,monkeypatch):
    root,deps,seen=fixture(tmp_path,monkeypatch);deps['prerequisite']=lambda *args:dict(passed=False)
    with pytest.raises(RuntimeError,match='prerequisite'):q.run(root,**deps)
    assert not (root/q.OUTPUT).exists() and not seen
