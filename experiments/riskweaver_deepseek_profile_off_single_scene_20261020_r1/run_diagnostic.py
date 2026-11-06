"""单场 DeepSeek 诊断：等待空闲 GPU，复用正式闭环，额外记录无凭据的调用正文。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
BASE = ROOT / 'experiments/riskweaver_deepseek_profile_off_single_scene_20261020_r1'
MODEL = 'deepseek-v4-pro'


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def wait_gpu():
    # 沿用排队偏好；不占用繁忙 GPU，且不修改其他用户的进程。
    while True:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        rows = [list(map(int, row.split(','))) for row in result.stdout.strip().splitlines()]
        available = [row for row in rows if row[0] in (1, 2, 3) and row[1] <= 10 and row[2] >= 14000]
        save(BASE / 'queue_status.json', {'state': 'ready' if available else 'waiting_for_idle_gpu',
             'checked_at': time.time(), 'gpu_rows_index_util_free_mib': rows,
             'criteria': 'GPU 1/2/3 utilization <=10%, free >=14000 MiB'})
        if available:
            return str(max(available, key=lambda row: row[2])[0])
        time.sleep(90)


def run():
    os.chdir(ROOT)
    gpu = wait_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    sys.path.insert(0, str(ROOT))
    from experiments.comparison import run_e1_group as runner
    from policies.llm_adversarial_planner import LLMAdversarialPlanner

    # 只覆盖此进程的配置，不修改公共运行器或正式方法源码。
    original_overrides = runner.group_overrides
    def overrides(*args, **kwargs):
        values = original_overrides(*args, **kwargs)
        return ['sim.llm.provider=deepseek' if value.startswith('sim.llm.provider=') else value for value in values]
    runner.group_overrides = overrides
    original_init = LLMAdversarialPlanner.__init__
    original_generate = LLMAdversarialPlanner.generate_attack_plan
    calls = []

    def append(record):
        with (BASE / 'llm_calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.provider != 'deepseek' or not self.available:
            raise RuntimeError('diagnostic_requires_configured_direct_deepseek')
        create = self.client.chat.completions.create
        def traced_create(*positional, **request):
            # 白名单记录消息；不序列化客户端、环境变量、请求头或异常正文。
            record = {'request_id': f'deepseek-scene182-{len(calls)+1:03d}',
                      'model_requested': request.get('model'), 'messages': request.get('messages'),
                      'started_at': time.time(), 'scene_step': getattr(self, '_diagnostic_step', None)}
            calls.append(record)
            start = time.monotonic()
            try:
                response = create(*positional, **request)
                message = response.choices[0].message
                record.update(status='response_received', model_returned=response.model,
                              content=message.content, usage=response.usage.model_dump() if response.usage else None,
                              finish_reason=response.choices[0].finish_reason)
                return response
            except Exception as error:
                record.update(status='service_error', error_type=type(error).__name__,
                              status_code=getattr(error, 'status_code', None))
                raise
            finally:
                record['elapsed_seconds'] = time.monotonic()-start
                append(record)
        self.client.chat.completions.create = traced_create

    def generate(self, env_state, *args, **kwargs):
        self._diagnostic_step = env_state.get('current_step', env_state.get('step'))
        count = len(calls)
        start = time.monotonic()
        plan = original_generate(self, env_state, *args, **kwargs)
        append({'event': 'local_validation', 'request_ids': [c['request_id'] for c in calls[count:]],
                'elapsed_including_validation_seconds': time.monotonic()-start,
                'plan_after_validation_global': plan, 'last_error': self.last_error,
                'service_failure': self.last_request_failed, 'planner_trace': self.last_trace})
        return plan

    LLMAdversarialPlanner.__init__ = init
    LLMAdversarialPlanner.generate_attack_plan = generate
    source = ROOT / 'experiments/e1_five_arm_idm_10_modelpool_20261014_r1/formal/riskweaver_profile_off/shard_007/manifest.json'
    manifest = json.loads(source.read_text())
    if manifest['original_scenario_indices'] != [182]:
        raise RuntimeError('unexpected_frozen_scene')
    save(BASE / 'manifest.json', manifest)
    save(BASE / 'launch.json', {'provider': 'deepseek', 'model': MODEL, 'profile_enabled': False,
         'obstacles_enabled': False, 'scenario_index': 182, 'seed': 42, 'maximum_steps': 400,
         'gpu': gpu, 'started_at': time.time(), 'selection_note': 'Previously interactive development scene; not representative benchmark',
         'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    sys.argv = ['run_e1_group.py', '--group', 'riskweaver_profile_off', '--manifest', str(BASE/'manifest.json'),
                '--output-dir', str(BASE/'run'), '--model', MODEL, '--models', MODEL,
                '--policy', 'idm', '--steps', '400', '--seed', '42']
    runner.main()


if __name__ == '__main__':
    try:
        run()
        save(BASE/'diagnostic_exit.json', {'exit_code': 0, 'finished_at': time.time()})
    except Exception as error:
        save(BASE/'diagnostic_exit.json', {'exit_code': 1, 'error_type': type(error).__name__, 'finished_at': time.time()})
        raise
