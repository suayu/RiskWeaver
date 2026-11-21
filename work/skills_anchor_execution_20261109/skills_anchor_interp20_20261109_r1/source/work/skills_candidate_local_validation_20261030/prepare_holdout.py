"""冻结原留出十初态；零API/零扩散，准备与测试输出分离，不据结果改选场景。"""
from pathlib import Path
import argparse,hashlib,json,os,random,shutil,sys,time
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_temporal_candidates_20261029'),str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from temporal_planner import TemporalCandidatePlanner
from model_pool_protocol import inspect_model_source
from launch_pool_pair import verify_sources

EXPECTED=[6,28,45,102,114,143,146,185,229,232]


def development_gate():
    run=ROOT/'experiments/skills_candidate_local_highlevel_development10_20261030_r4'
    result=json.loads((run/'results.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
    hashes=json.loads((run/'source_sha256.json').read_text());check=verify_sources(protocol,hashes)
    rows=result['results'];counts={mode:sum(r['accepted_plan'] for r in rows if r['mode']==mode) for mode in protocol['arms']}
    passed=bool(result['status']=='completed' and result['final_source_check']['verified'] and check['verified'] and len(rows)==30
        and all(sum(r['mode']==mode for r in rows)==10 for mode in protocol['arms'])
        and all(not r.get('runtime_failure') and not r.get('planner_service_failure') and r['budget']['transport_calls']<=4 for r in rows)
        and counts['temporal_candidate_local']>=6 and counts['temporal_candidate_local']>=counts['legacy_off'])
    return dict(passed=passed,counts=counts,current_source_verified=check['verified'],
        development_results_sha256=hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
        scope='initial_state_operational_gate_only; does_not_certify_all_prose_or_threat_or_full_episode')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists():raise ValueError('CPU_only_new_output_required')
    gate=development_gate()
    if not gate['passed']:raise ValueError('current_highlevel_development_gate_failed')
    source=ROOT/'work/skills_candidate_local_development_protocol_20261030_r4'
    manifest_path=ROOT/'work/skills_heldout_contract_protocol_20261029_r1/validation10.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest['original_scenario_indices']!=EXPECTED or manifest['scenario_count']!=10:raise ValueError('original_holdout_changed')
    for name,digest in zip(manifest['scenario_files'],manifest['scenario_sha256']):
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise ValueError('frozen_scenario_bytes_changed')
    dependencies=json.loads((source/'source_sha256.json').read_text());files=set(dependencies)
    files.update(f.relative_to(ROOT).as_posix() for f in HERE.glob('*.py'))
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(files)}
    a.output.mkdir();(a.output/'development_readiness.json').write_text(json.dumps(gate,indent=2))
    (a.output/'validation10.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    import torch
    from omegaconf import OmegaConf
    for name in ['PROJECT_ROOT','SCRATCH_ROOT']:os.environ.setdefault(name,str(ROOT))
    os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'));torch.set_num_threads(2)
    from simulator import Simulator
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from policies.evaluation_trace import plain_data
    from policies.difficulty_audit import json_safe
    cfg=OmegaConf.load(source/'resolved_config.yaml')
    cfg.sim.traffic_model.device='cpu';cfg.sim.traffic_model.multi_gpu_devices=[];cfg.sim.visualize=False;cfg.sim.save_carla_data=False
    cfg.sim.scenario_data_output_path=str(a.output/'initial_data');cfg.sim.movie_path=str(a.output/'unused_movies')
    def forbidden(*args,**kwargs):raise AssertionError('API_forbidden_while_preparing_holdout')
    LLMAdversarialPlanner._request_attack_plan=forbidden
    env=Simulator(cfg);env.test_files=manifest['scenario_files'];samples=[]
    flags=dict(extra_recipes=True,rolling_intent=True,explicit_contract=True,deduplicate=True,
        fractional_interaction=True,velocity_continuation=True,motion_evidence=True)
    for local,scene in enumerate(EXPECTED):
        seed=(42+1000003*scene)%(2**31-1);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        env.reset(local);state=env.get_state_for_planning()
        planner=object.__new__(LLMAdversarialPlanner);planner.allow_trajectory=True;planner.use_obstacles=False
        planner.last_request_failed=False;planner.last_trace={}
        engine=TemporalCandidatePlanner(cfg.sim,4,**flags);context=engine.build_context(env,state,planner)
        frame=dict(ego=env.ego_state,agents=env.data_dict['agent'][-1],active=env.agent_active)
        initial=json_safe(plain_data(dict(state=state,frame=frame)))
        f=a.output/f'initial_scene{scene:03d}.json';f.write_text(json.dumps(initial,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        samples.append(dict(scene=scene,scene_id=str(getattr(env,'current_scene_id',f'frozen-validation-{scene}')),source_step=0,
            state=initial['state'],frame=initial['frame'],source_trace=str(f),source_trace_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            contexts={'temporal_original':context},scenario_sha256=manifest['scenario_sha256'][local],effective_seed=seed))
        (a.output/'preparation_progress.json').write_text(json.dumps(dict(prepared=len(samples),planned=10,API_calls=0,env_steps=0)))
    inputs=a.output/'inputs.json';inputs.write_text(json.dumps(samples,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    prior=json.loads((source/'protocol.json').read_text())
    protocol=dict(prior,version='candidate_local_highlevel_original_holdout10_pair_20261030_r1',stage='validation',scenes=EXPECTED,
        arms=['legacy_off','temporal_candidate_local'],planned_state_arms=20,maximum_total_physical_requests=80,
        inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),model_source=inspect_model_source(ROOT/'policies/llm_adversarial_planner.py'),
        development_gate=gate,manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        scope='original_holdout_initial_states_only; not_project_unseen_or_full_episode',
        no_outcome_based_scene_selection=True,source_initial_inputs_sha256_unchanged=None,all_prose_correctness_certified=False)
    shutil.copyfile(source/'resolved_config.yaml',a.output/'resolved_config.yaml')
    (a.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    for name in hashes:
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    intact=verify_sources(protocol,hashes)
    (a.output/'stage_result.json').write_text(json.dumps(dict(status='completed',prepared=10,API_calls=0,env_steps=0,
        new_diffusion_samples=0,source_check=intact,development_gate=gate,goal_complete=False),indent=2))
    if not intact['verified']:raise ValueError('source_changed_during_holdout_preparation')
    print(json.dumps(dict(prepared=10,sources=len(hashes),API_calls=0,env_steps=0,source_verified=True)))


if __name__=='__main__':main()
