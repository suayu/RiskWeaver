"""新入口默认旧行为，显式开关留档，拒绝历史目录及旧冻结协议。"""
from pathlib import Path
import subprocess,sys,json
import pytest

ROOT=Path('/home2/zhaoyx/scenario-dreamer')

def command(tmp_path,skills=False):
    all_args=json.loads((ROOT/'work/skills_modelpool_paired_protocol_20261022_r2/prepared_arms.json').read_text())
    arm='skills_full_shared_recovery' if skills else 'legacy_shared_recovery'
    args=list(all_args[arm]['launch_command_pending_confirmation'][2:])
    args[args.index('--output-dir')+1]=str(tmp_path/'new-group')
    return [sys.executable,str(Path(__file__).with_name('run_recovery_v4.py')),*args]


def test_defaults_keep_legacy_mode_without_creating_files(tmp_path):
    result=subprocess.run(command(tmp_path)+['--prepare-only'],capture_output=True,text=True,check=True)
    p=json.loads(result.stdout)['recovery_v4_protocol']
    assert (p['execution_area_policy'],p['temporal_recovery'],p['route_tail_preflight'])==('legacy_relative','off','off')
    assert not (tmp_path/'new-group').exists()


def test_all_explicit_flags_remain_visible_in_prepared_protocol(tmp_path):
    result=subprocess.run(command(tmp_path,True)+['--execution-area-policy','positive_area','--temporal-recovery','on',
        '--temporal-validation-seconds','60','--route-tail-preflight','on','--video-export','integer','--prepare-only'],capture_output=True,text=True,check=True)
    value=json.loads(result.stdout);p=value['recovery_v4_protocol']
    assert p['execution_area_policy']=='positive_area' and p['temporal_recovery']==p['route_tail_preflight']=='on'
    assert value['video_export_protocol']['video_export']=='integer'
    assert p['temporal_acceptance_budget_seconds']==60
    assert not (tmp_path/'new-group').exists()


def test_tail_cannot_be_silently_enabled_for_skills_off_baseline(tmp_path):
    r=subprocess.run(command(tmp_path)+['--route-tail-preflight','on','--prepare-only'],capture_output=True,text=True)
    assert r.returncode!=0 and 'requires_explicit_skills' in r.stderr


def test_old_or_missing_source_protocol_is_rejected_before_real_inference(tmp_path):
    r=subprocess.run(command(tmp_path)+['--free-protection-confirmed'],capture_output=True,text=True)
    assert r.returncode!=0 and 'new_v4_source_manifest_required' in r.stderr
    assert not (tmp_path/'new-group').exists()


def test_existing_experiment_is_not_overwritten(tmp_path):
    p=tmp_path/'new-group';p.mkdir();marker=p/'results.json';marker.write_text('historical')
    r=subprocess.run(command(tmp_path)+['--prepare-only'],capture_output=True,text=True)
    assert r.returncode!=0 and 'preserve_existing_experiment' in r.stderr and marker.read_text()=='historical'
