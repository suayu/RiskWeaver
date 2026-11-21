"""原清单和四组配置不变，独立冻结共同提供方修复；准备阶段不调用API。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'work/skills_next_stage_protocol_20261026_r2'
OUT = ROOT/'work/skills_provider_controlled_protocol_20261028_r1'
EXP = ROOT/'experiments/skills_four_arm_provider_controlled_development10_20261028_r1'
sys.path.insert(0, str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from model_pool_protocol import inspect_model_source
from launch_pool_pair import verify_sources


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def set_flag(command, flag, value):
    if flag in command:
        command[command.index(flag)+1] = str(value)
    else:
        command.extend([flag, str(value)])


def main():
    if OUT.exists() or EXP.exists():
        raise FileExistsError('preserve_old_and_new_trials')
    metadata = json.loads((OLD/'protocol.json').read_text())
    old_hashes = json.loads((OLD/'source_sha256.json').read_text())
    if not verify_sources(metadata, old_hashes)['verified']:
        raise ValueError('prior_algorithm_changed_requires_review')
    confirmed = ROOT/metadata['free_protection_receipt']
    if not confirmed.exists():
        raise ValueError('existing_free_protection_receipt_missing')
    source = inspect_model_source(ROOT/'policies/llm_adversarial_planner.py')
    pool = [m for m in metadata['frozen_model_pool'] if m not in source['disabled_models']]
    if not pool or pool[0] != 'kimi-k3':
        raise ValueError('primary_model_unavailable_do_not_silently_replace_control')
    OUT.mkdir()
    for name in ['development10.json','validation10.json','smoke074.json']:
        original = OLD/name
        data = json.loads(original.read_text())
        for f,h in zip(data['scenario_files'], data['scenario_sha256']):
            if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h:
                raise ValueError('scenario_source_changed')
        (OUT/name).write_bytes(original.read_bytes())
    prepared = copy.deepcopy(json.loads((OLD/'prepared_arms.json').read_text()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LLM_MODEL_NAME=pool[0], LLM_MODEL_NAMES=','.join(pool))
    for arm, row in prepared.items():
        command = row['launch_command_pending_confirmation']
        command[1] = str(HERE/'run_controlled.py')
        set_flag(command, '--semantic-model-policy', 'service_only')
        set_flag(command, '--manifest', OUT/'development10.json')
        set_flag(command, '--output-dir', EXP/arm)
        start = command.index('--models')+1
        end = next(i for i in range(start,len(command)) if command[i].startswith('--'))
        command[start:end] = pool
        # 真实入口的prepare路径检查所有开关关系；该路径不构造客户端或发送HTTP。
        result = subprocess.run(command+['--prepare-only'], cwd=ROOT, env=env,
            capture_output=True, text=True, check=True)
        row.update(prepare_result=json.loads(result.stdout), started=False,
            semantic_model_policy='service_only', common_provider_control='v1')
    save(OUT/'prepared_arms.json', prepared)
    metadata.update(version='four_arm_common_provider_control_20261028_v1',
        original_protocol=str(OLD), status='prepared_not_started', frozen_model_pool=pool,
        model_source=source, launcher_entry=str(HERE/'launch_controlled.py'),
        provider_control=dict(exact_thinking_models=['glm-5.3','qwen3.8-2.4t-a95b'],
            semantic_rejection_model_rotation=False, genuine_service_fallback=True,
            no_extra_requests=True, root_legacy_defaults_unchanged=True),
        reused_smoke_scope='182/74 continuous and skills runtime unchanged; provider-only change has separate SDK budget regression',
        prior_failed_experiment_preserved='experiments/skills_four_arm_incremental_development10_20261026_r2')
    metadata['comparison_controls']['model_policy'] = 'kimi-k3 primary; no semantic-rejection rotation; approved pool only on genuine service/output failure; disclose all actual models'
    save(OUT/'protocol.json',metadata)
    names = set(old_hashes)
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.rglob('*')
        if p.is_file() and p.suffix in {'.py','.md','.json'} and '__pycache__' not in p.parts)
    names.update((OUT/n).relative_to(ROOT).as_posix() for n in ['development10.json','validation10.json','smoke074.json','prepared_arms.json','protocol.json'])
    hashes = {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    for n in hashes:
        p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    save(OUT/'source_sha256.json', hashes)
    save(OUT/'preparation_validation.json',dict(source_files=len(hashes), prepared_arms=len(prepared),
        same_development_manifest=(OUT/'development10.json').read_bytes()==(OLD/'development10.json').read_bytes(),
        same_validation_manifest=(OUT/'validation10.json').read_bytes()==(OLD/'validation10.json').read_bytes(),
        real_entry_preflight_checks=4, new_API_calls=0, experiments_started=0,
        source_verification=verify_sources(metadata,hashes)))
    print(json.dumps(dict(protocol=str(OUT),source_files=len(hashes),entry_checks=4,API_requests=0,experiments_started=0)))


if __name__=='__main__':
    main()
