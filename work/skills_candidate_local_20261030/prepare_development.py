"""原开发十初态不变；冻结三臂源码、相同免费模型池和120次物理调用上限。"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from model_pool_protocol import inspect_model_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=ROOT/'work/skills_highlevel_legacy_skills_protocol_20261030_r3'
    prior=json.loads((source/'protocol.json').read_text());files=set(json.loads((source/'source_sha256.json').read_text()))
    files.update(['policies/candidate_grounding.py','tests/test_candidate_local_grounding.py'])
    files.update(f.relative_to(ROOT).as_posix() for f in HERE.glob('*.py'))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(files)}
    if a.output.exists():raise FileExistsError('preserve_protocol')
    a.output.mkdir();shutil.copyfile(source/'inputs.json',a.output/'inputs.json');shutil.copyfile(source/'resolved_config.yaml',a.output/'resolved_config.yaml')
    protocol=dict(prior,version='candidate_local_highlevel_development10_three_arm_20261030_r4',
        arms=['legacy_off','temporal_global','temporal_candidate_local'],planned_state_arms=30,
        frozen_model_pool=['deepseek-v4-pro-0813','glm-5.3','qwen3.8-2.4t-a95b','qwen3.8-max','qwen3.7-flash','deepseek-v4.1-flash-1017'],
        model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),maximum_total_physical_requests=120,
        maximum_logical_initial_and_repair_requests=dict(legacy_off=1,temporal_global=2,temporal_candidate_local=2),
        source_initial_inputs_sha256_unchanged=hashlib.sha256((a.output/'inputs.json').read_bytes()).hexdigest()==prior['inputs_sha256'],
        global_identity_and_selected_candidate_original_checks_preserved=True,
        unselected_candidate_errors_isolated_only_in_local_arm=True,no_new_gate_or_execution=True,
        scope='same_development_initial_states_three_arms; no_event_grid_or_diffusion; correctness_requires_separate_review')
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'source_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    for n in hashes:
        dest=a.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    print(json.dumps(dict(sources=len(hashes),planned_state_arms=30,API_calls=0,inputs_unchanged=protocol['source_initial_inputs_sha256_unchanged'])))


if __name__=='__main__':main()
