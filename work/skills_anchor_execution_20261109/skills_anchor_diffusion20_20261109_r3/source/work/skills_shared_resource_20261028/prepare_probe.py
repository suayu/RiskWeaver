"""冻结两场40帧共享资源探测，原四组252源码与实验目录不变。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'work/skills_provider_controlled_protocol_20261028_r1'
OUT=ROOT/'work/skills_shared_resource_probe_protocol_20261028_r1'
EXP=ROOT/'experiments/skills_shared_resource_probe_20261028_r1'
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from launch_pool_pair import verify_sources
from model_pool_protocol import inspect_model_source


def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    if OUT.exists() or EXP.exists():raise FileExistsError('preserve_resource_probes')
    metadata=json.loads((OLD/'protocol.json').read_text());hashes=json.loads((OLD/'source_sha256.json').read_text())
    if not verify_sources(metadata,hashes)['verified']:raise ValueError('original_runtime_drift')
    manifest=json.loads((OLD/'development10.json').read_text());indices=[manifest['original_scenario_indices'].index(i) for i in [53,74]]
    selected=copy.deepcopy(manifest)
    for key in ['scenario_files','scenario_sha256','original_scenario_indices']:selected[key]=[manifest[key][i] for i in indices]
    selected.update(scenario_count=2,comparison_scene_count=2,selection_basis='predeclared_dense53_sparse74_resource_diagnostic_not_effectiveness')
    for f,h in zip(selected['scenario_files'],selected['scenario_sha256']):
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise ValueError('frozen_scene_changed')
    OUT.mkdir();save(OUT/'manifest.json',selected)
    command=list(json.loads((OLD/'prepared_arms.json').read_text())['skills_full_shared_recovery']['launch_command_pending_confirmation'])
    command[1]=str(HERE/'run_capped.py')
    for flag,value in [('--manifest',OUT/'manifest.json'),('--output-dir',EXP/'probe'),('--steps',40)]:command[command.index(flag)+1]=str(value)
    command+=['--resource-cap-gib','14','--resource-reserve-gib','3','--resource-fixture','on','--resource-observe-sampling','on']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LLM_MODEL_NAME=metadata['frozen_model_pool'][0],LLM_MODEL_NAMES=','.join(metadata['frozen_model_pool']))
    prepared=subprocess.run(command+['--prepare-only'],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
    protocol=dict(version='shared_resource_probe_v1_20261028',model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),
        frozen_model_pool=metadata['frozen_model_pool'],original_protocol=str(OLD),planned_resource_scenes=[53,74],
        max_frames_per_scene=40,cap_gib=14.,reserve_gib=3.,allowed_GPU_indices=['0','1'],max_concurrent_probe=1,
        max_wall_seconds=900,watchdog_min_free_gib=1.,watchdog_poll_seconds=2,
        deterministic_fixture=True,new_API_requests=0,not_attack_performance_evidence=True,
        user_authorization='2026-10-28 explicit reply: allow singleGPU shared resource probe, proceed only after pass',
        success_requires=['bounded_memory_no_OOM','two_source_scene_sampling_evidence','at_least10_actual_guided_frames',
            'at_least_one40frame_prefix','original8_joint_samples','no_runtime_or_sampling_or_wall_timeout_or_background_failure','trace_and_video_complete'],
        sampling_call_max_seconds=120,
        method_rejections_scope='retain_and_report_geometry_reachability_or_CPU_projection_rejections; not proof_of_GPU_memory_failure_or_method_success',
        original_solver_weights_and_safety_unchanged=True,command=command,prepare_result=json.loads(prepared.stdout),goal_complete=False)
    save(OUT/'protocol.json',protocol)
    names=set(hashes)
    names.update(p.relative_to(ROOT).as_posix() for p in HERE.glob('*') if p.is_file() and p.suffix in {'.py','.md','.json'})
    names.update((OUT/n).relative_to(ROOT).as_posix() for n in ['manifest.json','protocol.json'])
    frozen={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names)}
    for n in frozen:
        p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    save(OUT/'source_sha256.json',frozen)
    print(json.dumps(dict(protocol=str(OUT),source_files=len(frozen),real_prepare_checks=1,API_requests=0,experiments_started=0)))


if __name__=='__main__':main()
