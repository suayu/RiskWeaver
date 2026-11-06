"""验证新入口的公共视频开关、旧实现恢复及历史保护，禁止真实推理。"""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import pytest
from prospective_video_hook import install_integer_video_export
from test_integer_frame_export import fixture


def test_hook_rejects_historical_directory_before_modifying_runner(tmp_path):
    old=lambda **kw:'legacy';runner=NS(generate_video=old);root=tmp_path/'old';root.mkdir()
    with pytest.raises(FileExistsError):install_integer_video_export(root,runner)
    assert runner.generate_video is old


def test_new_run_integer_export_reaches_existing_audit_and_restores_old(tmp_path):
    root=tmp_path/'experiment';old=lambda **kw:'legacy';runner=NS(generate_video=old)
    restore=install_integer_video_export(root,runner)
    fixture(tmp_path,278);src=root/'scene'
    try:
        result=runner.generate_video(name='scenario_000',output_dir=src,delete_images=False)
        assert Path(result).is_file()
        from experiments.skills.audit import probe_video
        assert probe_video(Path(result))['frame_count']==278
        mapping=json.loads(Path(result).with_suffix('.frames.json').read_text())
        assert mapping['source_png_count']==278 and len(list(src.glob('*.png')))==278
        with pytest.raises(FileExistsError):runner.generate_video(name='scenario_000',output_dir=src,delete_images=False)
    finally:restore()
    assert runner.generate_video is old


def test_hook_rejects_external_output_or_image_deletion(tmp_path):
    root=tmp_path/'future';runner=NS(generate_video=lambda **kw:None);restore=install_integer_video_export(root,runner)
    try:
        with pytest.raises(ValueError,match='outside_new_run'):runner.generate_video(name='scene',output_dir=tmp_path/'other',delete_images=False)
        with pytest.raises(ValueError,match='preserve_source_images'):runner.generate_video(name='scene',output_dir=root/'scene',delete_images=True)
    finally:restore()


@pytest.mark.parametrize('mode',[None,'legacy','integer'])
def test_prepared_entry_has_same_real_runner_contract_and_no_outputs(tmp_path,mode):
    root=Path('/home2/zhaoyx/scenario-dreamer')
    prepared=json.loads((root/'work/skills_modelpool_paired_protocol_20261022_r2/prepared_arms.json').read_text())
    original=prepared['legacy_shared_recovery']['launch_command_pending_confirmation']
    args=list(original[2:]);args[args.index('--output-dir')+1]=str(tmp_path/'future-output')
    script=Path(__file__).with_name('run_with_video_export.py')
    result=subprocess.run([sys.executable,str(script),*(['--video-export',mode] if mode else []),*args,'--prepare-only'],capture_output=True,text=True,check=True)
    data=json.loads(result.stdout)
    assert data['video_export_protocol']['video_export']==(mode or 'legacy')
    assert data['prepare_only'] and data['skills']=='off'
    assert not (tmp_path/'future-output').exists()


def test_actual_entry_rejects_old_or_missing_source_freeze_before_inference(tmp_path):
    root=Path('/home2/zhaoyx/scenario-dreamer')
    prepared=json.loads((root/'work/skills_modelpool_paired_protocol_20261022_r2/prepared_arms.json').read_text())
    args=list(prepared['legacy_shared_recovery']['launch_command_pending_confirmation'][2:])
    args[args.index('--output-dir')+1]=str(tmp_path/'future-output')
    script=Path(__file__).with_name('run_with_video_export.py')
    result=subprocess.run([sys.executable,str(script),'--video-export','integer',*args,'--free-protection-confirmed'],capture_output=True,text=True)
    assert result.returncode!=0 and 'new_frozen_source_manifest_required' in result.stderr
    assert not (tmp_path/'future-output').exists()
