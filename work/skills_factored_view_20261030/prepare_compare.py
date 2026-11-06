"""冻结四个开发态的视图诊断；不是留出或原B阶段四知识层的替代品。"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from model_pool_protocol import inspect_model_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=ROOT/'work/skills_candidate_local_development_protocol_20261030_r4'
    prior=json.loads((source/'protocol.json').read_text());all_samples=json.loads((source/'inputs.json').read_text())
    samples=[s for s in all_samples if s['scene'] in [53,56,74,182]]
    if [s['scene'] for s in samples]!=[53,56,74,182]:raise ValueError('four_development_states_missing')
    if a.output.exists():raise FileExistsError('preserve_previous_diagnostic_protocol')
    a.output.mkdir();f=a.output/'inputs.json';f.write_text(json.dumps(samples,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    files=set(json.loads((source/'source_sha256.json').read_text()))
    files.update(f.relative_to(ROOT).as_posix() for f in HERE.glob('*.py'));files.add((HERE/'SKILL.md').relative_to(ROOT).as_posix())
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(files)}
    protocol=dict(version='lossless_factored_wire_four_development_states_20261030_r1',
        scenes=[53,56,74,182],arms=['original_wire_view','factored_wire_view'],planned_state_arms=8,
        frozen_model_pool=prior['frozen_model_pool'],model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),
        free_protection_receipt=prior['free_protection_receipt'],inputs_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
        physical_limit_per_state_arm=4,maximum_total_physical_requests=32,maximum_logical_requests=2,
        original_identity_candidates_validator_and_diffusion_unchanged=True,source_data_not_used_as_model_answer=True,
        scope='development_diagnostic; shared_default_view_and_reading_skill_package; not_full_skill_ablation_or_generalization',
        view_default='legacy',no_API_in_preparation=True,new_execution_frames=0,D=None,goal_complete=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'source_sha256.json').write_text(json.dumps(hashes,indent=2));shutil.copyfile(source/'resolved_config.yaml',a.output/'resolved_config.yaml')
    for n in hashes:
        dest=a.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dest)
    print(json.dumps(dict(frozen_sources=len(hashes),planned_state_arms=8,API_calls=0)))


if __name__=='__main__':main()
