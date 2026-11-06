"""独立的74单场排队器；保留冻结算法与旧实验，不自动派发规模实验。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
PROTOCOL = 'work/skills_next_stage_protocol_20261025_r1'
OUTPUT = 'experiments/skills_incremental_smoke074_20261025_r1'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def verify_files(root, protocol, hashes, inspect, transition):
    # 模型正常停用允许改变注释，其余冻结算法、知识和配置一律不可漂移。
    bad = [n for n, h in hashes.items() if n != 'policies/llm_adversarial_planner.py'
           and (not (root/n).is_file() or hashlib.sha256((root/n).read_bytes()).hexdigest() != h)]
    model = transition(protocol['model_source'], inspect(root/'policies/llm_adversarial_planner.py'))
    return dict(verified=not bad and model['verified'], changed_algorithm_files=bad, model_transition=model)


def run(root=ROOT, *, idle=None, inspect=None, transition=None, prerequisite=None,
        sleep=time.sleep, popen=subprocess.Popen, prepare=subprocess.run):
    out = root/OUTPUT
    # 最先拒绝旧目录，外层异常处理只允许写入本次成功创建的目录。
    if out.exists():
        raise FileExistsError('preserve_existing_single_smoke')
    if idle is None:
        sys.path[:0] = [str(root/'work/skills_modelpool_budget_dev_20261022'),
                       str(root/'work/skills_protocol_preparation_20261025')]
        from launch_pool_pair import idle_gpus
        from model_pool_protocol import inspect_model_source, verify_pool_transition
        from audit_four_arm import smoke_prerequisite
        idle, inspect, transition, prerequisite = idle_gpus, inspect_model_source, verify_pool_transition, smoke_prerequisite
    protocol_dir = root/PROTOCOL
    protocol = load(protocol_dir/'protocol.json')
    hashes = load(protocol_dir/'source_sha256.json')
    check = verify_files(root, protocol, hashes, inspect, transition)
    if not check['verified']:
        raise RuntimeError('frozen_source_drift_before_creation')
    previous = prerequisite(root/'experiments/skills_priority_v5_closed_loop_scene182_20261025_r1', 'priority_v5', 182)
    if not previous['passed']:
        raise RuntimeError('scene182_prerequisite_not_passed')
    if not (root/protocol['free_protection_receipt']).is_file():
        raise ValueError('existing_free_protection_confirmation_missing')
    manifest = load(protocol_dir/'smoke074.json')
    if manifest['original_scenario_indices'] != [74]:
        raise ValueError('wrong_smoke_scene')
    for n, h in zip(manifest['scenario_files'], manifest['scenario_sha256']):
        if hashlib.sha256(Path(n).read_bytes()).hexdigest() != h:
            raise ValueError('frozen_scene_changed')
    prepared = load(protocol_dir/'prepared_smoke.json')
    command = list(prepared['command'])
    if Path(command[command.index('--output-dir')+1]) != out/'incremental_full':
        raise ValueError('prepared_output_mismatch')
    command += ['--free-protection-confirmed']
    ready = prepare(command+['--prepare-only'], cwd=root, capture_output=True, text=True, check=True)
    prepare_result = json.loads(ready.stdout)
    if not prepare_result.get('free_protection_operator_confirmed'):
        raise ValueError('free_protection_flag_not_forwarded')
    if shutil.disk_usage(root).free < 5.5*2**30:
        raise RuntimeError('disk_below_5_5GiB_user_action_required')
    out.mkdir(exist_ok=False)
    try:
        out.with_suffix('.pid').write_text(str(os.getpid()))
        save(out/'manifest.json', manifest)
        # 冻结协议原件保持不动，仅在本次实验快照追加排队器自身来源。
        own = Path(__file__).resolve()
        own_name = own.relative_to(root).as_posix()
        hashes = dict(hashes, **{own_name: hashlib.sha256(own.read_bytes()).hexdigest()})
        save(out/'source_sha256.json', hashes)
        for n in hashes:
            destination = out/'source_snapshot'/n
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((root/n).read_bytes())
        save(out/'prepare_receipt.json', prepare_result)
        save(out/'launch.json', dict(controller_pid=os.getpid(), command=command,
            planned_attempts=1, scenario_index=74, maximum_physical_requests=4, maximum_frames=400,
            protocol=str(protocol_dir), source_files=len(hashes), prerequisite182=previous,
            free_protection_operator_confirmed=True, started_at=time.time(),
            automatic_next_stage=False, scope='incremental prototype real single smoke; not majority or superiority'))
        while True:
            if shutil.disk_usage(root).free < 5.5*2**30:
                raise RuntimeError('disk_below_5_5GiB_user_action_required')
            check = verify_files(root, protocol, hashes, inspect, transition)
            save(out/'runtime_source_integrity.json', check)
            if not check['verified']:
                raise RuntimeError('frozen_algorithm_changed_in_queue')
            available = idle(set())
            if available:
                gpu = available[0]
                break
            save(out/'queue_status.json', dict(state='waiting_for_completely_idle_gpu', checked_at=time.time(),
                no_existing_cuda_context=True, maximum_utilization=10, minimum_free_MiB=20000))
            sleep(90)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH='.:tests:safe-sim:safe-sim/trajdata/src',
                   LLM_MODEL_NAME=protocol['frozen_model_pool'][0], LLM_MODEL_NAMES=','.join(protocol['frozen_model_pool']))
        wrapped = ['bash', 'scripts/run_with_llm_provider.sh', 'dashscope', *command]
        save(out/'queue_status.json', dict(state='dispatching_on_completely_idle_gpu',gpu=gpu,checked_at=time.time()))
        with (out/'run.log').open('xb') as log:
            child = popen(wrapped, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            save(out/'method_pid.json', dict(pid=child.pid, gpu=gpu, command=wrapped, started_at=time.time()))
            code = child.wait()
        check = verify_files(root, protocol, hashes, inspect, transition)
        save(out/'runtime_source_integrity.json', check)
        if not check['verified']:
            raise RuntimeError('frozen_algorithm_changed_during_run')
        save(out/'launcher_result.json',dict(status='child_terminal',exit_code=code,finished_at=time.time(),
             automatic_next_stage=False, acceptance='requires_independent_trace_video_grounding_and_actual_threat_audit'))
        return code
    except Exception as error:
        # 只有上面mkdir成功后才可写终态；拒绝旧目录时不会进入此分支。
        save(out/'launcher_result.json', dict(status='launcher_failure', error_type=type(error).__name__,finished_at=time.time()))
        raise


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    sys.exit(run())
