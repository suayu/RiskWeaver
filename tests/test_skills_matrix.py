"""防止旧终态文件掩盖存活进程，或未通过冒烟便启动后续实验。"""
import json
from experiments.skills import launch_matrix as matrix
from experiments.skills import audit


def completed_group(attacks=6):
    return dict(artifacts_complete=True,runtime_failures=0,planner_service_failures=0,
                background_collision_frames=0,actual_attack_scenes=attacks)


def test_live_predecessor_has_priority_over_old_result(tmp_path,monkeypatch):
    directory=tmp_path/'prior';directory.mkdir()
    directory.with_suffix('.pid').write_text('12345')
    (directory/'launcher_result.json').write_text(json.dumps({'status':'completed'}))
    monkeypatch.setattr(matrix,'live',lambda pid:True)
    assert matrix.prerequisites(directory,'smoke')==(None,'predecessor_live')


def test_validation_needs_majority_and_artifacts(tmp_path,monkeypatch):
    directory=tmp_path/'prior';directory.mkdir()
    (directory/'launcher_result.json').write_text(json.dumps({'status':'completed'}))
    groups={'legacy':completed_group(3),'skills':completed_group(0),'skills_expanded':completed_group(5)}
    monkeypatch.setattr(audit,'audit_experiment',lambda path:dict(all_terminal=True,groups=groups))
    assert matrix.prerequisites(directory,'development')==(False,'insufficient_actual_attack_scenes')
    groups['skills_expanded']['actual_attack_scenes']=6
    assert matrix.prerequisites(directory,'development')==(True,'passed')
    groups['legacy']['artifacts_complete']=False
    assert matrix.prerequisites(directory,'development')[0] is False


def test_safety_failure_does_not_pass_coverage_gate(tmp_path,monkeypatch):
    directory=tmp_path/'prior';directory.mkdir()
    (directory/'launcher_result.json').write_text(json.dumps({'status':'completed'}))
    group=completed_group(10);group['background_collision_frames']=1
    monkeypatch.setattr(audit,'audit_experiment',lambda path:dict(all_terminal=True,groups={'skills_expanded':group}))
    assert matrix.prerequisites(directory,'development')==(False,'predecessor_integrity_or_runtime_failure:skills_expanded')
