"""冻结共同混合指令修复版本；复用原两场配置，不调用API或启动GPU。"""
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'work/skills_hybrid_closed_loop2_protocol_20261102_r1'
TARGET = ROOT / 'work/skills_hybrid_closed_loop2_clean_protocol_20261102_r2'
EXPERIMENT = ROOT / 'experiments/skills_hybrid_closed_loop2_clean_paired_20261102_r2'
sys.path.insert(0, str(ROOT / 'work/skills_modelpool_budget_dev_20261022'))
from launch_pool_pair import verify_sources
from model_pool_protocol import inspect_model_source


def retarget(protocol, target, experiment, root=ROOT):
    """只替换入口和输出位置；同模型/清单/预算/安全不能悄然改变。"""
    value = copy.deepcopy(protocol)
    if value['planned_scenes'] != [6, 45] or value['planned_attempts'] != 4:
        raise ValueError('unexpected_source_scene_scope')
    if value['models'] != ['qwen3.8-2.4t-a95b']:
        raise ValueError('unexpected_source_model')
    if value['maximum_physical_SDK_per_scene'] != 4:
        raise ValueError('unexpected_source_budget')
    for arm in value['arms']:
        command = value['commands'][arm]
        if command[1] != str(root / 'work/hybrid_closed_loop_20261102/run_rag_worker.py'):
            raise ValueError('unexpected_old_worker')
        command[1] = str(root / 'work/hybrid_instruction_scope_20261102/run_worker.py')
        for flag, replacement in [('--manifest', target / 'manifest.json'),
                                  ('--output-dir', experiment / arm)]:
            command[command.index(flag) + 1] = str(replacement)
    value.update(version='hybrid_closed_loop2_clean_instruction_v2_20261102',
                 experiment_root=str(experiment), predecessor_experiment=protocol['experiment_root'],
                 instruction_scope_version='hybrid_instruction_scope_v1',
                 actual_SDK_instruction_delivery_verified=False, goal_complete=False)
    return value


def main():
    if TARGET.exists() or EXPERIMENT.exists():
        raise FileExistsError('preserve_existing_protocol_and_results')
    base = json.loads((BASE / 'protocol.json').read_text())
    previous = Path(base['experiment_root'])
    if json.loads((previous / 'launcher_result.json').read_text())['status'] != 'completed':
        raise ValueError('predecessor_not_terminal')
    hashes = json.loads((BASE / 'source_sha256.json').read_text())
    check = verify_sources(base, hashes)
    if not check['verified']:
        raise ValueError(('base_source_changed', check))
    protocol = retarget(base, TARGET, EXPERIMENT)
    model_source = inspect_model_source(ROOT / 'policies/llm_adversarial_planner.py')
    if any(model not in model_source['active_models'] for model in protocol['models']):
        raise ValueError('frozen_model_no_longer_active')
    protocol['model_source'] = model_source
    extra = [p for p in HERE.glob('*.py') if not p.name.startswith('source_')]
    names = set(hashes) | {p.relative_to(ROOT).as_posix() for p in extra}
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    manifest = (BASE / 'manifest.json').read_bytes()
    if hashlib.sha256(manifest).hexdigest() != protocol['manifest_sha256']:
        raise ValueError('base_manifest_changed')
    TARGET.mkdir()
    (TARGET / 'manifest.json').write_bytes(manifest)
    (TARGET / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (TARGET / 'source_sha256.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    for name in hashes:
        dest = TARGET / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    receipt = dict(status='prepared_not_started', planned_attempts=4, scenes=[6, 45],
                   frozen_sources=len(hashes), API_requests=0, training_started=False,
                   same_solver_safety_and_samples=True, instruction_changed_for_both_arms=True)
    (TARGET / 'preparation_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
