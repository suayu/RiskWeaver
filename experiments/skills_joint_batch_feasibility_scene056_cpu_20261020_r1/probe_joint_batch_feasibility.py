"""固定失败场景原计划，重建同一CPU样本批并逐一校验；不调用LLM或执行仿真。"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
from types import SimpleNamespace as NS

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
os.environ.setdefault('PROJECT_ROOT',str(ROOT));os.environ.setdefault('SCRATCH_ROOT',str(ROOT))
os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'))
import numpy as np
import torch
from omegaconf import OmegaConf
from simulator import Simulator
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.ego_profile import JointTrajectorySafetyGate
from policies.trajectory_projection import project_positions
from policies.risk_metrics import RouteReachability
from policies.joint_safety import NoSafeJointCandidate

BASE=ROOT/'experiments/skills_v2_frozen_development10_cpu_pair_20261020_r1/skills_v2'
OUT=ROOT/'experiments/skills_joint_batch_feasibility_scene056_cpu_20261020_r1'


def seeded(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU isolated probe only')
    OUT.mkdir(exist_ok=False)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    source=BASE/'movies/scenario_056/execution_trace.jsonl'
    rows=[json.loads(line) for line in source.open()]
    old=next(x['joint'] for x in rows if x['kind']=='proposal_prediction' and x['step']==0)
    state=next(x for x in rows if x['kind']=='state' and x['step']==0)
    plan=next(x['validated_plan'] for x in rows if x['kind']=='llm_output' and x['step']==0)
    manifest=json.loads((BASE.parent/'manifest.json').read_text())
    scene_file=manifest['scenario_files'][manifest['original_scenario_indices'].index(56)]
    cfg=OmegaConf.load(BASE/'resolved_config.yaml')
    cfg.sim.visualize=False;cfg.sim.scenario_data_output_path=str(OUT/'scenario_debug')
    cfg.sim.traffic_model.device='cpu'
    def forbidden(*a,**k):raise AssertionError('LLM_network_forbidden_in_fixed_batch_probe')
    LLMAdversarialPlanner._request_attack_plan=forbidden
    torch.set_num_threads(2);seeded(42)
    env=Simulator(cfg);env.test_files=[scene_file]
    seeded((42+1000003*56)%(2**31-1));env.reset(0)
    initial=np.asarray(env.data_dict['agent'][-1],float)
    if not np.array_equal(initial,np.asarray(state['agents'],float)) or not np.array_equal(env.ego_state,np.asarray(state['ego'])):
        raise RuntimeError('initial_physical_state_not_exactly_replayed')
    env.diffusion_controller.update_iterative_guidance(dict(cfg.sim.traffic_model.guidance.params),[1.5,1.,2.,4.])
    env.set_attack_intent(target_id=plan['attack_target_id'],anchors=plan['anchors'],strategy=plan['strategy'])
    env.attack_intent.update(duration=plan['duration'],request_id=plan['request_id'])
    os.environ['RISKWEAVER_JOINT_DIAGNOSTIC_DIR']=str(OUT/'samples')
    preparation_error=None
    try:env.prepare_background_traffic()
    except NoSafeJointCandidate as error:preparation_error=str(error)
    files=list((OUT/'samples').glob('*.npz'))
    if len(files)!=1:raise RuntimeError('expected_one_frozen_joint_batch')
    batch=np.load(files[0]);positions=batch['positions'];ids=batch['agent_ids'];selected=int(batch['selected_index'])
    original=np.asarray(old['positions_global'],float)
    difference=float(np.abs(positions[selected]-original).max())
    # 原始回放允许浮点舍入容差；若不等价仍可保留诊断，但不能声称原批次被精确重建。
    exact_reconstruction=bool(difference<=1e-5 and np.array_equal(ids,np.asarray(old['agent_ids'])))
    limits=JointTrajectorySafetyGate(cfg.sim).dynamic_limits;settings=cfg.sim.traffic_model.dynamics_projection
    record=dict(scene=56,source_trace=str(source),source_trace_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_config_sha256=hashlib.sha256((BASE/'resolved_config.yaml').read_bytes()).hexdigest(),
        source_plan=plan,selected_index=selected,original_selected_index=old['metadata']['joint_sample_index'],
        selected_max_difference_from_original_m=difference,exact_batch_reconstruction=exact_reconstruction,
        initial_state_equal=True,original_preparation_rejection=preparation_error,limits=limits,
        max_deviation_m=float(settings.max_deviation_m),max_seconds_per_vehicle=float(settings.max_seconds_per_vehicle),
        LLM_calls=0,actual_executed_frames=0,scope='fixed-intent native CPU batch feasibility probe only; not closed-loop generation coverage',candidates=[])
    for sample in range(1,len(positions)):
        candidate=dict(sample_index=sample,selected_by_original=sample==selected,projection_rejections=[],actors=[])
        pp=[];vv=[];yy=[]
        for row,agent_id in enumerate(ids):
            actor=dict(agent_id=int(agent_id));started=time.monotonic()
            try:
                p,v,y,audit=project_positions(positions[sample,row],initial[agent_id],.1,limits,
                    float(settings.max_seconds_per_vehicle),float(settings.max_deviation_m))
                actor.update(status='passed',audit=audit);pp.append(p);vv.append(v);yy.append(y)
            except NoSafeJointCandidate as error:
                actor.update(status='rejected',reason=str(error));candidate['projection_rejections'].append(str(error))
            actor['elapsed_seconds']=time.monotonic()-started;candidate['actors'].append(actor)
        if not candidate['projection_rejections']:
            joint=NS(agent_ids=ids,positions_global=np.asarray(pp),velocities_global=np.asarray(vv),
                     yaws_global=np.asarray(yy),valid_mask=np.ones((len(ids),positions.shape[2]),bool))
            replay=NS(data_dict=env.data_dict,agent_active=env.agent_active,ego_state=env.ego_state,
                      scenario_dict=env.scenario_dict,pending_joint_trajectory=joint,dt=.1,
                      previously_attacked_agent_ids=(),get_static_obstacles=env.get_static_obstacles)
            try:
                JointTrajectorySafetyGate(cfg.sim).validate_joint(replay)
                candidate['joint_background_and_physics_pass']=True
                reach=RouteReachability(cfg.sim.evaluation.reachability)
                base=reach.compute(replay,False,[]);danger=reach.compute(replay,True,[])
                candidate.update(baseline_area_m2=base['area_m2'],dangerous_area_m2=danger['area_m2'],
                                 positive_area_pass=base['area_m2']>0 and danger['area_m2']>0)
            except NoSafeJointCandidate as error:
                candidate.update(joint_background_and_physics_pass=False,joint_rejection=str(error))
            np.savez_compressed(OUT/f'projected_sample_{sample}.npz',positions_global=joint.positions_global,
                                raw_positions_global=positions[sample],agent_ids=ids,initial_states=initial)
        record['candidates'].append(candidate)
        (OUT/'probe_result.partial.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    record['feasible_samples']=[c['sample_index'] for c in record['candidates'] if c.get('joint_background_and_physics_pass') and c.get('positive_area_pass')]
    record['finished_at']=time.time()
    (OUT/'probe_result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    (OUT/'exit_status.json').write_text(json.dumps(dict(exit_code=0)))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():(OUT/'exit_status.json').write_text(json.dumps(dict(exit_code=1,error_type=type(error).__name__)))
        raise
