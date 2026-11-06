"""冻结原方案四组开发/留出配置与单场74冒烟命令；零API，不启动实验。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from model_pool_protocol import inspect_model_source

OUT = ROOT/'work/skills_next_stage_protocol_20261025_r1'
EXPERIMENT = ROOT/'experiments/skills_four_arm_incremental_development10_20261025_r1'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def set_flag(command, flag, value):
    if flag in command:
        command[command.index(flag)+1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    if OUT.exists() or EXPERIMENT.exists():
        raise FileExistsError('preserve_previous_protocol_and_trials')
    old = ROOT/'work/skills_recovery_paired_protocol_20261023_v4'
    if not (ROOT/'work/bailian_pool_protection_confirmed_20261023.json').is_file():
        raise ValueError('existing_free_protection_confirmation_required')
    models = json.loads((old/'protocol.json').read_text())['frozen_model_pool']
    metadata = inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    active = [m for m in models if m not in metadata['disabled_models']]
    pool = (['kimi-k3'] if 'kimi-k3' in active else [])+[m for m in active if m != 'kimi-k3']
    if not pool:
        raise ValueError('approved_free_pool_empty')
    preparation = json.loads((ROOT/'work/skills_incremental_entry_checks_20261025/actual_entry_prepare_final.json').read_text())
    original = json.loads((old/'development10.json').read_text())
    validation = json.loads((old/'validation10.json').read_text())
    assert not set(original['original_scenario_indices']) & set(validation['original_scenario_indices'])
    for manifest in [original, validation]:
        for name, digest in zip(manifest['scenario_files'], manifest['scenario_sha256']):
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    OUT.mkdir(exist_ok=False)
    (OUT/'development10.json').write_bytes((old/'development10.json').read_bytes())
    (OUT/'validation10.json').write_bytes((old/'validation10.json').read_bytes())
    smoke = copy.deepcopy(original)
    index = original['original_scenario_indices'].index(74)
    for key in ['scenario_files','scenario_sha256','original_scenario_indices']:
        smoke[key] = [original[key][index]]
    smoke.update(scenario_count=1, comparison_scene_count=1, selection_basis='diagnostic_scene74_before_scale_not_generalization')
    save(OUT/'smoke074.json', smoke)
    arms = {
        'legacy_shared_recovery':dict(skills_enabled=False, level='none', candidate_expansion=False, active_recipes=False),
        'catalog_no_knowledge':dict(skills_enabled=True, level='none', candidate_expansion=False, active_recipes=False),
        'skills_no_expansion':dict(skills_enabled=True, level='full', candidate_expansion=False, active_recipes=False),
        'skills_full_shared_recovery':dict(skills_enabled=True, level='full', candidate_expansion=True, active_recipes=True),
    }
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LLM_MODEL_NAME=pool[0], LLM_MODEL_NAMES=','.join(pool))
    prepared = {}
    for arm, flags in arms.items():
        template = next(c for c in preparation['checks'] if c['mode'] == ('on' if flags['skills_enabled'] else 'off') and c['recipes'] == 'off')
        command = list(template['command'])
        command.remove('--prepare-only')
        for flag, value in [('--manifest',OUT/'development10.json'), ('--output-dir',EXPERIMENT/arm),
                            ('--incremental-skills','on' if flags['skills_enabled'] else 'off'),
                            ('--active-window-recipes','on' if flags['active_recipes'] else 'off'),
                            ('--skill-level',flags['level']), ('--candidate-expansion','on' if flags['candidate_expansion'] else 'off'),
                            ('--model',pool[0])]:
            set_flag(command,flag,value)
        start=command.index('--models')+1
        end=next(i for i in range(start,len(command)) if command[i].startswith('--'))
        command[start:end]=pool
        tested=subprocess.run(command+['--prepare-only'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        prepared[arm]=dict(launch_command_pending_confirmation=command, prepare_result=json.loads(tested.stdout), started=False)
    command=list(prepared['skills_full_shared_recovery']['launch_command_pending_confirmation'])
    set_flag(command,'--manifest',OUT/'smoke074.json')
    set_flag(command,'--output-dir',ROOT/'experiments/skills_incremental_smoke074_20261025_r1/incremental_full')
    smoke_test=subprocess.run(command+['--prepare-only'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    save(OUT/'prepared_smoke.json',dict(command=command,prepare_result=json.loads(smoke_test.stdout),started=False))
    save(OUT/'prepared_arms.json',prepared)
    save(OUT/'protocol.json',dict(version='prospective_four_arm_incremental_20261025',status='prepared_not_started',
        arms=arms,frozen_model_pool=pool,model_source=metadata,free_protection_receipt='work/bailian_pool_protection_confirmed_20261023.json',
        planned_development_attempts=40,validation_scope='same original leaveout10; not whole-project-unseen; not automatically started',
        required_before_scale=['existing_v5_smoke_terminal_and_audited','incremental_smoke074_valid_actual_execution_and_artifacts'],
        required_smoke_receipts=[
            dict(experiment='experiments/skills_priority_v5_closed_loop_scene182_20261025_r1',arm='priority_v5',scenario_index=182),
            dict(experiment='experiments/skills_incremental_smoke074_20261025_r1',arm='incremental_full',scenario_index=74)],
        comparison_controls=dict(P1_strict_threshold=6,all_attempts_in_denominator=True,common_provider_adapter='observed',
            same_solver=[.2,.5,2],same_joint_samples=8,same_attack_duration=10,same_physical_request_limit=4,
            common_recovery='positive_area + same-batch recovery + at-most-one-frame whole-joint tail recovery + integer video',
            B_vs_C='same deterministic catalog/evidence/protocol; only loaded skill documents differ',
            C_vs_D='candidate expansion plus new active-window recipes; cannot call this a skill-text effect',
            A_vs_D='whole pipeline package comparison; screening schedule differs and must be disclosed',
            model_policy='approved free pool automatic fallback; report mixtures; mixed models cannot prove skill-text causality',
            no_continuous_refinement_changes=True,all_four_groups_must_be_audited=True),
        automatic_start=False,goal_complete=False))
    names=set(json.loads((ROOT/'experiments/skills_priority_v5_closed_loop_scene182_20261025_r1/source_sha256.json').read_text()))
    for folder in [ROOT/'work/skills_incremental_evidence_20261025',HERE]:
        names.update(p.relative_to(ROOT).as_posix() for p in folder.rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix in {'.py','.md','.json'})
    for name in ['development10.json','validation10.json','smoke074.json','prepared_arms.json','prepared_smoke.json','protocol.json']:
        names.add((OUT/name).relative_to(ROOT).as_posix())
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    for name in hashes:
        destination=OUT/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes((ROOT/name).read_bytes())
    save(OUT/'source_sha256.json',hashes)
    print(json.dumps(dict(protocol=str(OUT),source_files=len(hashes),entry_prepare_checks=5,API_requests=0,experiments_started=0)))


if __name__=='__main__':main()
