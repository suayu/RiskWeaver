"""入口默认关闭、参数透传及异常清理；不调用API或预训练网络。"""
import importlib.util,sys
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('refined_entry',Path(__file__).with_name('run_refined.py'))
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def test_disabled_entry_passes_args_and_restores_process_globals(tmp_path,monkeypatch):
    script=tmp_path/'original.py';output=tmp_path/'arguments.txt'
    script.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(sys.argv[2])\n')
    monkeypatch.setattr(sys,'argv',['entry',str(script),str(output),'unchanged'])
    argv=sys.argv;paths=list(sys.path);entry.main()
    assert output.read_text()=='unchanged' and sys.argv is argv and sys.path==paths


def test_enabled_entry_exception_restores_real_sampler_and_joint_selector(tmp_path,monkeypatch):
    from tbsim.models.diffusion import DiffusionTraj
    from policies.diffusion_model_wrapper import DiffusionModelWrapper
    before=DiffusionTraj.sample,DiffusionModelWrapper._select_joint_guided_action
    script=tmp_path/'failed.py';script.write_text('raise RuntimeError("original_script_failure")\n')
    monkeypatch.setattr(sys,'argv',['entry','--refinement','proximal',str(script)])
    argv=sys.argv;paths=list(sys.path)
    with pytest.raises(RuntimeError,match='original_script_failure'):entry.main()
    assert before==(DiffusionTraj.sample,DiffusionModelWrapper._select_joint_guided_action)
    assert sys.argv is argv and sys.path==paths
