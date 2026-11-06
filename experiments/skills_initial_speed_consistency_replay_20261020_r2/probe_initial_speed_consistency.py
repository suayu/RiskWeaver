"""离线审计初态与速度门槛矛盾；观测速度上界仅为研究分支，默认逻辑不变。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT))
import numpy as np
from omegaconf import OmegaConf
from policies.ego_profile import JointTrajectorySafetyGate
from policies.trajectory_projection import project_positions
from policies.joint_safety import NoSafeJointCandidate
from policies.risk_metrics import RouteReachability

OUT=ROOT/'experiments/skills_initial_speed_consistency_replay_20261020_r2'
SOURCES={
    'gpu_catalog_053': ROOT/'experiments/skills_stage_d_development10_bailian_20261020_r2/catalog_expanded',
    'cpu_skills_056': ROOT/'experiments/skills_v2_frozen_development10_cpu_pair_20261020_r1/skills_v2',
}
SCENES={'gpu_catalog_053':53,'cpu_skills_056':56}


def initial_speed_certificate(initial,limits,dt):
    """三角不等式给出首步所需最小减速度；超过上限即可证明原约束无解。"""
    speed=float(np.linalg.norm(initial[2:4]))
    cap=min(limits['max_speed'],limits['max_step_distance']/dt)
    required=max(0.,(speed-cap)/dt)
    normals=np.column_stack((np.cos(np.arange(8)*np.pi/8),np.sin(np.arange(8)*np.pi/8)))
    factor=np.cos(np.pi/16)*(1-1e-6)
    support_need=float(max(0.,np.abs(normals@initial[2:4]).max()-cap*factor))
    return dict(initial_speed_mps=speed,first_step_speed_cap_mps=cap,
                minimum_deceleration_required_mps2=required,max_acceleration_mps2=limits['max_acceleration'],
                norm_infeasible=required>limits['max_acceleration']+1e-8,
                polygon_infeasible=bool(support_need>limits['max_acceleration']*dt*factor+1e-8),
                support_velocity_change_required_mps=support_need)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU isolated replay only')
    OUT.mkdir(exist_ok=False)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    results=[]
    for case,base in SOURCES.items():
        cfg=OmegaConf.load(base/'resolved_config.yaml');settings=cfg.sim.traffic_model.dynamics_projection
        limits=JointTrajectorySafetyGate(cfg.sim).dynamic_limits
        scene=SCENES[case];trace=base/f'movies/scenario_{scene:03d}/execution_trace.jsonl'
        rows=[json.loads(line) for line in trace.open()]
        prop=next(x for x in rows if x['kind']=='proposal_prediction')
        state=next(x for x in rows if x['kind']=='state' and x['step']==prop['step'])
        scene_input=next(x for x in rows if x['kind']=='llm_input' and x['step']==prop['step'])['state']
        states=np.asarray(state['agents'],float)
        record=dict(case=case,source_trace=str(trace),source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                    original_limits=limits,actors=[],LLM_calls=0,actual_executed_frames=0,
                    scope='initial-speed consistency counterfactual; not deployed, not actual execution or generation success')
        positions=[];velocities=[];yaws=[];rejections=[]
        for index,agent_id in enumerate(prop['joint']['agent_ids']):
            initial=states[agent_id];reference=np.asarray(prop['joint']['positions_global'][index],float)
            local_limits=copy.deepcopy(limits);speed=float(np.linalg.norm(initial[2:4]))
            local_limits['max_speed']=max(local_limits['max_speed'],speed)
            local_limits['max_step_distance']=max(local_limits['max_step_distance'],speed*.1)
            actor=dict(agent_id=agent_id,original_infeasibility_certificate=initial_speed_certificate(initial,limits,.1),
                       counterfactual_limits=local_limits)
            try:
                p,v,y,audit=project_positions(reference,initial,.1,local_limits,
                    float(settings.max_seconds_per_vehicle),float(settings.max_deviation_m))
                actor.update(status='passed_counterfactual_projection',audit=audit)
                positions.append(p);velocities.append(v);yaws.append(y)
            except NoSafeJointCandidate as error:
                actor.update(status='rejected',reason=str(error));rejections.append(actor['reason'])
                # 仅为解释保留超出0.25m时所需的位移，不能让其进入安全审计。
                if str(error)=='projection_excessive_deviation':
                    try:
                        *_,diagnostic=project_positions(reference,initial,.1,local_limits,2.,1e6)
                        actor['diagnostic_only_unbounded_deviation']=diagnostic
                    except NoSafeJointCandidate as second:actor['diagnostic_only_error']=str(second)
            record['actors'].append(actor)
        record['projection_rejections']=rejections
        if not rejections:
            joint=NS(agent_ids=np.asarray(prop['joint']['agent_ids']),positions_global=np.asarray(positions),
                     velocities_global=np.asarray(velocities),yaws_global=np.asarray(yaws),
                     valid_mask=np.asarray(prop['joint']['valid_mask'],bool))
            env=NS(pending_joint_trajectory=joint,data_dict={'agent':[states]},agent_active=np.asarray(state['active'],bool),
                   ego_state=np.asarray(state['ego'],float),dt=.1,previously_attacked_agent_ids=(),
                   get_static_obstacles=lambda:state['static_obstacles'],scenario_dict={'route':scene_input['route']})
            gate=JointTrajectorySafetyGate(cfg.sim)
            # 每辆车已经通过各自更严格的上界；原门控继续核对背景OBB/中点/静态与加速度jerk。
            gate.dynamic_limits=dict(limits,max_speed=max(a['counterfactual_limits']['max_speed'] for a in record['actors']),
                                     max_step_distance=max(a['counterfactual_limits']['max_step_distance'] for a in record['actors']))
            try:
                gate.validate_joint(env);record['background_and_physics_gate_pass']=True
                reach=RouteReachability(cfg.sim.evaluation.reachability)
                record['baseline_area']=reach.compute(env,False,state['static_obstacles'])
                record['dangerous_area']=reach.compute(env,True,state['static_obstacles'])
            except NoSafeJointCandidate as error:
                record.update(background_and_physics_gate_pass=False,remaining_rejection=str(error))
            np.savez_compressed(OUT/f'{case}_projected.npz',positions_global=joint.positions_global,
                                raw_positions_global=np.asarray(prop['joint']['positions_global']),
                                initial_states=states,agent_ids=joint.agent_ids)
        results.append(record)
        (OUT/'probe_result.partial.json').write_text(json.dumps(dict(results=results),ensure_ascii=False,indent=2))
    (OUT/'probe_result.json').write_text(json.dumps(dict(results=results,finished_at=time.time(),LLM_calls=0,actual_executed_frames=0),ensure_ascii=False,indent=2))
    (OUT/'exit_status.json').write_text(json.dumps(dict(exit_code=0)))


if __name__=='__main__':main()
