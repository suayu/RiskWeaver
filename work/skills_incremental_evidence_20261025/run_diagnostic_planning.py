"""事先固定两负例一正例的六单元规划诊断；不是随机测试集或闭环效果评估。"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import copy
import hashlib
import json
import os
import sys
import time
import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT/'work/skills_priority_v5_20261024'),
               str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from incremental_planner import IncrementalSkillsPlanner
from model_pool_protocol import inspect_model_source, verify_pool_transition
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from bailian_transport import install_provider_compatibility


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('planning_diagnostic_requires_hidden_GPU')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    output = parser.parse_args().output_dir
    if output.exists():
        raise FileExistsError('preserve_previous_diagnostic')
    prior = ROOT/'experiments/skills_priority_v5_frozen_development10_pair_20261024_r1'
    original = json.loads((prior/'inputs.json').read_text())
    inputs = [next(s for s in original if s['scenario_index'] == index) for index in [74, 207, 220]]
    protocol = ROOT/'work/skills_recovery_paired_protocol_20261023_v4'
    approved = json.loads((protocol/'protocol.json').read_text())['frozen_model_pool']
    if not (ROOT/'work/bailian_pool_protection_confirmed_20261023.json').is_file():
        raise ValueError('existing_free_protection_receipt_required')
    before = inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    names = set(json.loads((prior/'launch.json').read_text())['source_sha256'])
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix in {'.py', '.md', '.json'})
    hashes = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    output.mkdir(exist_ok=False)
    output.with_suffix('.pid').write_text(str(os.getpid()))
    for name in hashes:
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    save(output/'source_sha256.json', hashes)
    save(output/'inputs.json', inputs)
    save(output/'launch.json', dict(arms=['contrast_only', 'contrast_with_recipes'], planned=6,
        source_model_before=before, preferred_model='kimi-k3', approved_models=approved, source_sha256=hashes,
        selection_rule='diagnostic_negative74_and220_positive207_from_completed_initial_state_audit',
        scope='development diagnostic only; no majority, independent-generalization or actual-threat claims',
        maximum_physical_requests_per_arm_state=4, maximum_published_candidates=24, started_at=time.time()))
    records = []
    restore = install_provider_compatibility()
    try:
        for number, item in enumerate(inputs):
            order = ['contrast_only', 'contrast_with_recipes'] if number % 2 == 0 else ['contrast_with_recipes', 'contrast_only']
            for arm in order:
                active = [m for m in approved if m not in inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')['disabled_models']]
                pool = (['kimi-k3'] if 'kimi-k3' in active else [])+[m for m in active if m != 'kimi-k3']
                if not pool:
                    raise RuntimeError('approved_free_pool_exhausted')
                os.environ['LLM_MODEL_NAME'] = pool[0]
                os.environ['LLM_MODEL_NAMES'] = ','.join(pool)
                cfg = OmegaConf.load(ROOT/'experiments/skills_modelpool_recovery_development10_paired_20261023_r3/skills_full_shared_recovery/resolved_config.yaml')
                request = copy.deepcopy(item['request'])
                state = request['state'];frame = item['frame'];route = np.asarray(state['route'], float)
                env = NS(ego_state=np.asarray(frame['ego'], float), data_dict={'agent':[np.asarray(frame['agents'], float)]},
                    agent_active=np.asarray(frame['active'], bool), current_step=request['step'],
                    current_scene_id=f"scenario_{item['scenario_index']:03d}", scenario_dict={'route':route, 'lanes':[route]},
                    dt=float(cfg.sim.dt), get_static_obstacles=lambda:[], cfg=cfg, pending_joint_trajectory=None)
                env.get_state_for_planning = lambda state=state:state
                planner = LLMAdversarialPlanner(model_name=pool[0], model_names=pool, provider='dashscope', attack_mode='trajectory_only')
                if not planner.available:
                    raise RuntimeError('bailian_not_available')
                planner.client = planner.client.with_options(max_retries=0)
                physical = []
                original_create = planner.client.chat.completions.create

                def create(*args, **kwargs):
                    if len(physical) >= 4:
                        raise RuntimeError('physical_request_budget_exhausted')
                    event = dict(arm=arm, scenario_index=item['scenario_index'], model_requested=kwargs.get('model'),
                                 messages=kwargs.get('messages'), extra_body=kwargs.get('extra_body'))
                    physical.append(event);started = time.monotonic()
                    try:
                        response = original_create(*args, **kwargs)
                        event.update(model_returned=response.model, content=response.choices[0].message.content,
                                     usage=response.usage.model_dump() if response.usage else None)
                        return response
                    except Exception as error:
                        event.update(error_type=type(error).__name__, status_code=getattr(error, 'status_code', None))
                        raise
                    finally:
                        event['elapsed_seconds'] = time.monotonic()-started
                        with (output/'api_calls.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(event, ensure_ascii=False)+'\n')

                planner.client.chat.completions.create = create
                engine = IncrementalSkillsPlanner(cfg.sim, 4, extra_recipes=arm == 'contrast_with_recipes')
                started = time.monotonic()
                plan = engine.plan(planner, env, state, request.get('instruction') or ['使用动态离散攻击挑战自车，禁止静态障碍物。'])
                record = dict(arm=arm, scenario_index=item['scenario_index'], plan=plan, trace=planner.last_trace,
                    accepted=bool(plan and plan.get('attack')), physical_requests=len(physical),
                    planner_service_failure=planner.last_request_failed, elapsed_seconds=time.monotonic()-started,
                    actual_attack_frames=0, D=None)
                records.append(record)
                save(output/f"scene_{item['scenario_index']:03d}_{arm}.json", record)
                save(output/'results.partial.json', dict(planned=6, attempted=len(records), unattempted=6-len(records), records=records))
                if planner.last_request_failed:
                    raise RuntimeError('final_planner_service_failure')
    finally:
        restore()
        unchanged = all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == h for n, h in hashes.items()
                        if n != 'policies/llm_adversarial_planner.py')
        transition = verify_pool_transition(before, inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'))
        save(output/'results.json', dict(planned=6, attempted=len(records), unattempted=6-len(records), records=records,
             source_unchanged=unchanged, model_transition=transition, actual_execution_frames=0, D=None,
             scope='diagnostic selected states; not an attack-coverage experiment', goal_complete=False))
    code = 0 if len(records) == 6 and unchanged and transition['verified'] and all(r['accepted'] for r in records) else 1
    save(output/'exit_status.json', dict(exit_code=code, finished_at=time.time()))
    print(json.dumps(dict(attempted=len(records), accepted=sum(r['accepted'] for r in records), source_unchanged=unchanged)))


if __name__ == '__main__':main()
