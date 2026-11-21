"""只准备一个74号真实重规划任务；不派发，不复用历史结果作为本版成功。"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from model_pool_protocol import inspect_model_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--experiment-name',default='skills_candidate_local_closed_loop_scene074_20261030_r2');a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve_smoke_protocol')
    if Path(a.experiment_name).name!=a.experiment_name or not a.experiment_name.startswith('skills_candidate_local_'):raise ValueError('explicit_workspace_experiment_basename_required')
    source=ROOT/'work/skills_candidate_local_holdout_protocol_20261030_r1'
    validation=ROOT/'experiments/skills_candidate_local_highlevel_holdout10_20261030_r1'
    result=json.loads((validation/'results.json').read_text());rows=result['results']
    if result['status']!='completed' or len(rows)!=20 or not result['final_source_check']['verified']:
        raise ValueError('current_highlevel_validation_not_complete')
    counts={mode:sum(r['accepted_plan'] for r in rows if r['mode']==mode) for mode in ['legacy_off','temporal_candidate_local']}
    if counts['temporal_candidate_local']<6 or counts['temporal_candidate_local']<counts['legacy_off']:
        raise ValueError('highlevel_initial_operational_majority_not_reached')
    original=ROOT/'experiments/skills_temporal_development10_paired_20261029_r1'
    manifest=json.loads((original/'manifest.json').read_text());i=manifest['original_scenario_indices'].index(74)
    manifest=dict(manifest,scenario_count=1,comparison_scene_count=1,original_scenario_indices=[74],
        scenario_files=[manifest['scenario_files'][i]],scenario_sha256=[manifest['scenario_sha256'][i]])
    if hashlib.sha256(Path(manifest['scenario_files'][0]).read_bytes()).hexdigest()!=manifest['scenario_sha256'][0]:raise ValueError('scenario_changed')
    a.output.mkdir();(a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    cmd=json.loads((original/'skills_full_shared_recovery_pid.json').read_text())['command'][3:]
    cmd[1]=str(HERE/'run_worker.py');cmd[cmd.index('--manifest')+1]=str(a.output/'manifest.json')
    output=ROOT/'experiments'/a.experiment_name/'temporal_local'
    cmd[cmd.index('--output-dir')+1]=str(output)
    pool=['deepseek-v4-pro-0813','glm-5.3','qwen3.8-2.4t-a95b','qwen3.8-max','qwen3.7-flash','deepseek-v4.1-flash-1017']
    start=cmd.index('--models')+1;end=start
    while end<len(cmd) and not cmd[end].startswith('--'):end+=1
    cmd[start:end]=pool;cmd[cmd.index('--model')+1]=pool[0]
    cmd+=['--highlevel-grounding','candidate_local','--identity-comparison-facts','on']
    prepared=subprocess.run(cmd+['--prepare-only'],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src'),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
    if prepared.returncode:raise RuntimeError('actual_worker_prepare_failed')
    data=json.loads(prepared.stdout);(a.output/'actual_prepare.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    files=set(json.loads((source/'source_sha256.json').read_text()))
    files.update(f.relative_to(ROOT).as_posix() for f in HERE.glob('*.py'))
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(files)}
    prior=json.loads((source/'protocol.json').read_text())
    # 实际闭环协议重新构造，不能继承原留出十初态的场景/臂数/输入哈希等元数据。
    protocol=dict(version='candidate_local_actual_single074_20261030_r2',
        stage='actual_replanning_single_scene',scenes=[74],planned_scenes=[74],arms=['temporal_local'],planned_attempts=1,
        manifest_sha256=hashlib.sha256((a.output/'manifest.json').read_bytes()).hexdigest(),
        free_protection_receipt=prior['free_protection_receipt'],
        highlevel_validation_results_sha256=hashlib.sha256((validation/'results.json').read_bytes()).hexdigest(),
        scope='actual_replanning_development_scene074; not_initial_state_pair_or_independent_generalization',
        model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),frozen_model_pool=pool,
        command=cmd,worker_output=str(output),maximum_total_physical_requests=4,highlevel_validation_counts=counts,
        allow_shared_V100S=True,allowed_GPU_indices=['0','1','2','3'],minimum_free_gib=17,own_cap_gib=14,reserve_gib=3,
        maximum_own_GPU_workers=1,previous_shared_probe='experiments/skills_shared_resource_probe_20261028_r1',
        no_diffusion_refinement_change=True,highlevel_and_diffusion_outcomes_separate=True,goal_complete=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    (a.output/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    for name in hashes:
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    print(json.dumps(dict(sources=len(hashes),scene=74,prepare_passed=True,new_API_calls=0,new_execution_frames=0)))


if __name__=='__main__':main()
