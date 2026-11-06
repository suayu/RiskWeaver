"""冻结下轮开发/独立验证协议；不调用模型、不执行场景、不改历史快照。"""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from model_pool_protocol import inspect_model_source

OUT=ROOT/'work/skills_recovery_paired_protocol_20261023_v4'
PREVIOUS=ROOT/'work/skills_modelpool_paired_protocol_20261022_r2'
EXPERIMENT=ROOT/'experiments/skills_modelpool_recovery_development10_paired_20261023_r2'


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    if OUT.exists() or EXPERIMENT.exists():raise FileExistsError('preserve_previous_protocol_and_experiment')
    prepared=json.loads((PREVIOUS/'prepared_arms.json').read_text())
    metadata=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    models=json.loads((PREVIOUS/'protocol.json').read_text())['frozen_model_pool']
    if not any(m not in metadata['disabled_models'] for m in models):raise ValueError('frozen_pool_all_disabled')
    manifests={name:json.loads((PREVIOUS/f'{name}10.json').read_text()) for name in ('development','validation')}
    ids=[set(m['original_scenario_indices']) for m in manifests.values()]
    assert not ids[0]&ids[1]
    for m in manifests.values():
        assert m['scenario_count']==10
        for path,digest in zip(m['scenario_files'],m['scenario_sha256']):assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    proof=json.loads((Path(__file__).with_name('recorded_component_verification.json')).read_text())
    assert proof['source_unchanged'] and len(proof['checks'])==2 and all(x['passed'] for x in proof['checks'])
    OUT.mkdir(exist_ok=False)
    for name in manifests:
        # 字节一致复制，不能按失败/成功结果改动原冻结开发与独立验证场景。
        (OUT/f'{name}10.json').write_bytes((PREVIOUS/f'{name}10.json').read_bytes())
    for arm,entry in prepared.items():
        command=entry['launch_command_pending_confirmation']
        command[1]=str(Path(__file__).with_name('run_recovery_v4.py'))
        command[command.index('--manifest')+1]=str(OUT/'development10.json')
        command[command.index('--output-dir')+1]=str(EXPERIMENT/arm)
        command+=['--video-export','integer','--execution-area-policy','positive_area','--temporal-recovery','on',
                  '--temporal-validation-seconds','60','--route-tail-preflight','on' if arm=='skills_full_shared_recovery' else 'off']
        tested=subprocess.run(command+['--prepare-only'],cwd=ROOT,capture_output=True,text=True,check=True)
        entry.update(prepare_result=json.loads(tested.stdout),started=False)
    sources=[ROOT/n for n in json.loads((PREVIOUS/'source_sha256.json').read_text())]
    sources.extend([ROOT/'policies/attack_execution_gate.py',ROOT/'utils/viz.py'])
    for folder in ['work/skills_video_export_fix_20261023','work/skills_recovery_v4_20261023','work/skills_audit_v2_dev_20261020']:
        sources.extend((ROOT/folder).rglob('*.py'))
    hashes={}
    for p in sorted(set(sources)):
        if '__pycache__' in p.parts:continue
        relative=p.relative_to(ROOT);data=p.read_bytes();dest=OUT/'source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        hashes[relative.as_posix()]=hashlib.sha256(data).hexdigest()
    dump(OUT/'source_sha256.json',hashes);dump(OUT/'prepared_arms.json',prepared)
    protocol=copy.deepcopy(json.loads((PREVIOUS/'protocol.json').read_text()))
    protocol.update(version='prospective-skills-recovery-v4-20261023',status='prepared_not_started',model_source=metadata,
        free_protection_confirmation='all10_operator_confirmed_20261023',
        free_protection_receipt='work/bailian_pool_protection_confirmed_20261023.json',
        previous_failed_trial_preserved='experiments/skills_modelpool_development10_paired_20261023_r1',
        changes=dict(common_video='integer_exact_trace_frames',common_execution_area_policy='positive_area',
            common_temporal_joint_recovery=dict(enabled=True,max_consecutive=1,acceptance_seconds=60,
                no_extrapolation=True,all_active_agents=True,requires_completed_previous_transition=True,
                recheck_original_dynamics_and_background_and_positive_area=True),
            skills_route_tail_preflight='nominal route-following state at end10-frame validity; no route extension or future ground truth'),
        temporal_validation_cost_evidence=dict(first30_second_acceptance='failed_timeout_not_success',
            diagnostic_budget=120,actual_full_validation_seconds=30.698619712144136,
            next_common_budget=60,note='cost separately recorded; does not change inner_lr,inner_beta,n_guide_steps, safety limits or four-request budget'),
        resource_policy='completely idle GPU only; no CUDA context, util<=10%, free>=20GB; disk>=5.5GiB before launch',
        conclusion_scope='same original development10 after disclosed engineering fixes; cannot count as independent validation or skill-text causal evidence')
    dump(OUT/'protocol.json',protocol)
    print(json.dumps(dict(protocol=str(OUT),experiment=str(EXPERIMENT),source_files=len(hashes),started=False,API_calls=0),ensure_ascii=False))


if __name__=='__main__':main()
