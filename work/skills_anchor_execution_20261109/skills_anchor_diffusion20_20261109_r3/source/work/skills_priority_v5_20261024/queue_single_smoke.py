"""仅运行一次新版技能闭环冒烟；保留历史，等待完全空闲GPU，不自动扩展实验。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from launch_pool_pair import idle_gpus
from model_pool_protocol import inspect_model_source, verify_pool_transition

OUT = ROOT/'experiments/skills_priority_v5_closed_loop_scene182_20261025_r1'


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    old = ROOT/'work/skills_recovery_paired_protocol_20261023_v4'
    if OUT.exists():
        raise FileExistsError('preserve_existing_smoke')
    if not (ROOT/'work/bailian_pool_protection_confirmed_20261023.json').is_file():
        raise ValueError('existing_free_protection_receipt_missing')
    paired = ROOT/'experiments/skills_priority_v5_frozen_development10_pair_20261024_r1'
    terminal = json.loads((paired/'results.json').read_text())
    if terminal['attempted'] != 20 or not terminal['source_unchanged'] or terminal['final_planner_service_failure']:
        raise ValueError('planning_contract_check_not_completed')
    before = inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    original_pool = json.loads((old/'protocol.json').read_text())['frozen_model_pool']
    active = [m for m in original_pool if m not in before['disabled_models']]
    pool = (['kimi-k3'] if 'kimi-k3' in active else []) + [m for m in active if m != 'kimi-k3']
    if not pool:
        raise ValueError('approved_free_pool_empty')
    source_names = set(json.loads((old/'source_sha256.json').read_text()))
    source_names.update(p.relative_to(ROOT).as_posix() for p in HERE.rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py', '.md', '.json'})
    # 冒烟沿用已完成规划配对的算法；只允许额外加入本排队器与模型正常停用。
    prior = json.loads((paired/'launch.json').read_text())['source_sha256']
    if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() != h for n, h in prior.items()):
        raise ValueError('planning_pair_algorithm_drift')
    hashes = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(source_names)}
    original = json.loads((old/'development10.json').read_text())
    position = original['original_scenario_indices'].index(182)
    manifest = copy.deepcopy(original)
    for key in ['scenario_files', 'scenario_sha256', 'original_scenario_indices']:
        manifest[key] = [original[key][position]]
    manifest.update(scenario_count=1, comparison_scene_count=1,
                    selection_basis='diagnostic_scene182_same_as_contract_probe_not_generalization')
    if hashlib.sha256(Path(manifest['scenario_files'][0]).read_bytes()).hexdigest() != manifest['scenario_sha256'][0]:
        raise ValueError('frozen_scene_changed')
    prepared = json.loads((ROOT/'work/skills_priority_v5_validation_20261024/actual_entry_prepare_checks.json').read_text())
    command = list(next(c for c in prepared['checks'] if c['arm'] == 'skills_full_shared_recovery')['command'])
    command.remove('--prepare-only')
    for flag, value in [('--manifest', OUT/'manifest.json'), ('--output-dir', OUT/'priority_v5')]:
        command[command.index(flag)+1] = str(value)
    start = command.index('--models')+1
    end = next(i for i in range(start, len(command)) if command[i].startswith('--'))
    command[start:end] = pool
    command[command.index('--model')+1] = pool[0]
    command.append('--free-protection-confirmed')
    # 在独立work目录提供真实单场清单，prepare-only不创建实验或发起请求。
    proof = ROOT/'work/skills_priority_v5_single_smoke_prepare_20261025_r1'
    proof.mkdir(exist_ok=False)
    save(proof/'manifest.json', manifest)
    prepare_command = list(command)
    prepare_command[prepare_command.index('--manifest')+1] = str(proof/'manifest.json')
    tested = subprocess.run(prepare_command+['--prepare-only'], cwd=ROOT, capture_output=True, text=True, check=True)
    prepare_result = json.loads(tested.stdout)
    if shutil.disk_usage(ROOT).free < 5.5*2**30:
        raise RuntimeError('disk_below_5_5GiB_user_action_required')
    OUT.mkdir(exist_ok=False)
    OUT.with_suffix('.pid').write_text(str(os.getpid()))
    save(OUT/'manifest.json', manifest)
    save(OUT/'source_sha256.json', hashes)
    for name in hashes:
        destination = OUT/'source_snapshot'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    save(OUT/'prepare_receipt.json', prepare_result)
    save(OUT/'launch.json', dict(controller_pid=os.getpid(), command=command, models=pool,
         planned_attempts=1, scenario_index=182, maximum_frames=400, maximum_physical_requests=4,
         model_source=before, source_files=len(hashes), started_at=time.time(),
         priority='correct understanding and actual execution before scale',
         scope='single dynamic/profile-off smoke; not majority or threat superiority evidence',
         automatic_next_stage=False))

    def integrity():
        bad = [n for n, h in hashes.items() if n != 'policies/llm_adversarial_planner.py'
               and hashlib.sha256((ROOT/n).read_bytes()).hexdigest() != h]
        transition = verify_pool_transition(before, inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'))
        result = dict(verified=not bad and transition['verified'], changed_algorithm_files=bad, model_transition=transition)
        save(OUT/'runtime_source_integrity.json', result)
        if not result['verified']:
            raise ValueError('algorithm_changed_after_freeze')

    while True:
        if shutil.disk_usage(ROOT).free < 5.5*2**30:
            raise RuntimeError('disk_below_5_5GiB_user_action_required')
        integrity()
        available = idle_gpus(set())
        if available:
            gpu = available[0]
            break
        save(OUT/'queue_status.json', dict(state='waiting_for_completely_idle_gpu',
             no_existing_cuda_context=True, maximum_utilization=10, minimum_free_MiB=20000, checked_at=time.time()))
        time.sleep(90)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONPATH='.:tests:safe-sim:safe-sim/trajdata/src',
               LLM_MODEL_NAME=pool[0], LLM_MODEL_NAMES=','.join(pool))
    wrapped = ['bash', 'scripts/run_with_llm_provider.sh', 'dashscope', *command]
    with (OUT/'run.log').open('xb') as log:
        process = subprocess.Popen(wrapped, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT)
        save(OUT/'method_pid.json', dict(pid=process.pid, gpu=gpu, command=wrapped, started_at=time.time()))
        code = process.wait()
    integrity()
    save(OUT/'launcher_result.json', dict(status='child_terminal', exit_code=code, finished_at=time.time(),
         automatic_next_stage=False, acceptance='requires_independent_trace_and_video_audit'))
    return code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        if OUT.exists():
            save(OUT/'launcher_result.json', dict(status='launcher_failure', error_type=type(error).__name__, finished_at=time.time()))
        raise
