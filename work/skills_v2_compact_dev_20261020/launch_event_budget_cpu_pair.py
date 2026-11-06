"""预定开发场景80的CPU配对；不共享GPU，不改已运行矩阵。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
NAME = 'skills_event_budget_cpu_paired_scene080_20261020_r2'
OUT = ROOT/'experiments'/NAME
PYTHON = '/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python'


def dump(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=False)
    (ROOT/'experiments'/f'{NAME}.pid').write_text(str(os.getpid())+'\n')
    source = ROOT/'work/skills_validation_protocol_20261020_v1_waitfix'
    protocol = json.loads((source/'protocol.json').read_text())
    # 场景80在本轮开发集；选择依据是旧trace存在较晚候选，不声称未见验证。
    manifest = json.loads((ROOT/'experiments/skills_stage_d_development10_bailian_20261020_r2/manifest.json').read_text())
    index = manifest['original_scenario_indices'].index(80)
    selected = dict(manifest)
    selected['name'] = 'skills_event_budget_development_scene080'
    selected['original_scenario_indices'] = [80]
    selected['scenario_count'] = selected['comparison_scene_count'] = 1
    selected['selection_basis'] = 'development late-opportunity diagnostic; not held-out'
    selected['scenario_files'] = [manifest['scenario_files'][index]]
    # 原清单中的逐文件哈希字段必须与选中的文件同步，其他元数据保留。
    for key in ['scenario_hashes', 'scenario_sha256']:
        if isinstance(manifest.get(key), list):
            selected[key] = [manifest[key][index]]
    dump('manifest.json', selected)
    dev = ROOT/'work/skills_v2_compact_dev_20261020'
    tracked = [p for p in dev.rglob('*') if p.is_file() and p.suffix in {'.py', '.md', '.json'}]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    dump('launch.json', dict(
        version='v2-event-budget-cpu-development', source_sha256=hashes,
        scene=80, selection='development late-opportunity diagnostic; not a held-out success claim',
        model='qwen3.8-flash', provider='dashscope', profile=False, dynamic_only=True,
        seed=42, steps=400, device='cpu', torch_threads=2, max_api_calls=4,
        arms=['fixed_opportunities', 'actual_api_budget'], max_seconds_per_arm=1800,
        compact_view=True, candidate_expansion=True, solver=protocol['solver'],
        attack_duration_frames=10, started_at=time.time(),
        speed_scope='CPU-vs-CPU comparison only; no GPU latency inference'))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               LLM_MODEL_NAME='qwen3.8-flash', LLM_MODEL_NAMES='qwen3.8-flash',
               PYTHONPATH='.:tests:safe-sim:safe-sim/trajdata/src:'+str(dev))
    results = []
    for arm, event in [('fixed_opportunities', 'off'), ('actual_api_budget', 'on')]:
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() != h for n, h in hashes.items()):
            raise RuntimeError('development source drift before arm start')
        cmd = ['nice', '-n', '10', 'bash', 'scripts/run_with_llm_provider.sh', 'dashscope', PYTHON,
               str(dev/'run_closed_loop.py'), '--device', 'cpu', '--compact-view', 'on',
               '--event-budget', event, '--skills', 'on', '--candidate-expansion', 'on',
               '--skill-level', 'full', '--group', 'riskweaver_profile_off',
               '--manifest', str(OUT/'manifest.json'), '--output-dir', str(OUT/arm),
               '--model', 'qwen3.8-flash', '--models', 'qwen3.8-flash',
               '--policy', 'idm', '--seed', '42', '--steps', '400']
        with (OUT/f'{arm}.log').open('w') as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            dump(f'{arm}_pid.json', dict(pid=process.pid, started_at=time.time()))
            timed_out = False
            try:
                code = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                # 只终止本启动器直接创建的有界开发子进程，保留全部已产生数据。
                timed_out = True
                process.terminate()
                try:
                    code = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
        results.append(dict(arm=arm, exit_code=code, timed_out=timed_out))
        dump('launcher_result.partial.json', dict(groups=results))
        attempts = list((OUT/arm).rglob('attempt_result.json'))
        if any(json.loads(p.read_text()).get('planner_service_failures', 0) for p in attempts):
            dump('launcher_result.json', dict(groups=results, status='planner_service_failure', finished_at=time.time()))
            return
    dump('launcher_result.json', dict(groups=results,
        status='completed' if all(g['exit_code'] == 0 for g in results) else 'completed_with_failures',
        finished_at=time.time()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            dump('launcher_result.json', dict(status='launcher_failure', error_type=type(error).__name__, finished_at=time.time()))
        raise
