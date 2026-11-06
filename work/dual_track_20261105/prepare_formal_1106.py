"""冻结新的配对协议，不改正在运行的短程验证或已停止的实验。"""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')


def main():
    old=ROOT/'work/dual_track_formal_protocol_20261105_r1'
    new=ROOT/'work/skills_dual_track_paired20_protocol_20261106_r1'
    new.mkdir(exist_ok=False)
    protocol=json.loads((old/'protocol.json').read_text())
    files=json.loads((old/'source_sha256.json').read_text())
    for name in ['work/dual_track_20261105/run_dual_worker_v4.py','work/dual_track_20261105/audit_dual.py','work/dual_track_20261105/scene_fault_policy.py','work/dual_track_20261105/test_scene_fault_policy.py','work/dual_track_20261105/test_empty_fix.py']:
        files[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in files:
        target=new/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    shutil.copyfile(old/'manifest.json',new/'manifest.json')
    protocol.update(experiment_root=str(ROOT/'experiments/skills_dual_track_paired20_20261106_r1'),
        source_count=len(files),experiment_started=False,
        change_scope='same-goal Skills package off/on; equal static permissions; explicit parallel dual evaluation; original generation safety unchanged',
        supersedes_experiment=str(ROOT/'experiments/skills_kimi_acceptance20_20261105_r1'),
        smoke_experiment=str(ROOT/'experiments/dual_track_smoke_20261105_r1'),
        smoke_scope='metric wiring only; shared instruction corrected for formal fairness; not performance evidence',
        source_protocol=str(new),scoring_status='exploratory_uncalibrated_reference_proxy',
        ability_formula='100*I*(1 if safe_completed else .5*progress); I excludes actual collision/off_route',
        score_comparison_scope='same IDM; conditional observed-state challenge, not universal cross-policy ability ranking')
    protocol['single_scene_fault_policy']='continue_next_fixed_index; exclude_system_faults_from_effect_stats; disclose_full_fault_ledger; normal_no_attack_retained'
    protocol['supersedes_experiment']=str(ROOT/'experiments/skills_dual_track_paired20_20261105_r3')
    protocol['dual_track'].update(route_corridor_half_width_m=4,control_library_size=144,
        minimum_progress_ratio=.2,maximum_speed_mps=25,maximum_lateral_speed_mps=3,
        weights=[.5,.3,.2],reference_library_empty='invalid_null_not_proven_unsolvable')
    for arm,command in protocol['commands'].items():
        command[1]=str(ROOT/'work/dual_track_20261105/run_dual_worker_v4.py')
        command[command.index('--manifest')+1]=str(new/'manifest.json')
        command[command.index('--output-dir')+1]=str(Path(protocol['experiment_root'])/arm)
    for name,data in [('protocol.json',protocol),('source_sha256.json',files)]:
        (new/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
    # 两组用真实参数解析链准备，不请求API或执行场景。
    for arm,command in protocol['commands'].items():
        with (new/(arm+'_prepare.log')).open('xb') as log:
            status=subprocess.run(command+['--prepare-only'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if status.returncode:raise RuntimeError(arm+'_prepare_failed')
    print(json.dumps(dict(protocol=str(new),sources=len(files),planned=protocol['planned_attempts'],prepare='both_passed')))


if __name__=='__main__':main()
