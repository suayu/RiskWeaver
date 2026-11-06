"""同状态只读复算：CV基线、记录联合未来、无交通反事实；零LLM/扩散采样。"""
from pathlib import Path
from types import SimpleNamespace as NS
from dataclasses import replace
import hashlib
import json
import os
import pickle
import sys
import time
import numpy as np
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
from policies.risk_metrics import RouteReachability
from policies.traffic_types import JointTrajectory
from policies.ego_profile import JointTrajectorySafetyGate
from policies.trajectory_projection import project_joint
from policies.joint_safety import validate_execution_prefix


def decode(value):
    return JointTrajectory(source_step=value['source_step'],agent_ids=np.asarray(value['agent_ids'],int),
        positions_global=np.asarray(value['positions_global'],float),velocities_global=np.asarray(value['velocities_global'],float),
        yaws_global=np.asarray(value['yaws_global'],float),valid_mask=np.asarray(value['valid_mask'],bool),metadata=value['metadata'])


def replay_env(cfg,scene,row,joint,previous):
    return NS(cfg=cfg,data_dict={'agent':[np.asarray(row['agents'],float)]},agent_active=np.asarray(row['active'],bool),
        ego_state=np.asarray(row['ego'],float),scenario_dict=scene,dt=.1,current_step=row['step'],pending_joint_trajectory=joint,
        previously_attacked_agent_ids=previous,get_static_obstacles=lambda:row['static_obstacles'],
        get_scenario_frame=lambda:NS(static_obstacles_global=np.asarray([[*s['center'],s['yaw'],s['length'],s['width']] for s in row['static_obstacles']],float).reshape(-1,5)))


def area(reach,env,joint_mode,empty=False):
    original=env.agent_active.copy()
    if empty:env.agent_active[:]=False
    try:r=reach.compute(env,use_joint_trajectory=joint_mode,static_obstacles=[] if empty else env.get_static_obstacles())
    finally:env.agent_active=original
    return {k:r.get(k) for k in ['area_m2','solvable','area_curve','backend','horizon_seconds']}


def main():
    assert os.getenv('CUDA_VISIBLE_DEVICES')==''
    source=ROOT/'experiments/skills_modelpool_development10_paired_20261023_r1/skills_full_shared_recovery'
    output=ROOT/'work/skills_rejection_replay_20261023/results.json'
    if output.exists():raise FileExistsError(output)
    cfg=OmegaConf.load(source/'resolved_config.yaml');records=[]
    for p in sorted(source.rglob('attempt_result.json')):
        a=json.loads(p.read_text());index=a['scenario_index']
        if index not in [56,74,181]:continue
        with Path(a['scenario_source']).open('rb') as stream:scene=pickle.load(stream)
        rows=[json.loads(line) for line in p.with_name('execution_trace.jsonl').read_text().splitlines()]
        states={r['step']:r for r in rows if r['kind']=='state'}
        end=a['executed_steps'];previous=sorted({r['target_id'] for r in rows if r['kind']=='plan_accepted' and r['step']<=end})
        reach=RouteReachability(cfg.sim.evaluation.reachability);results=[]
        for step in [end-1,end]:
            row=states[step];pred=next((r['joint'] for r in reversed(rows) if r['kind']=='execution_prediction' and r['step']==step),None)
            joint=decode(pred) if pred else None;env=replay_env(cfg,scene,row,joint,previous)
            route,arc,tangent=reach._route_geometry(scene['route']);nearest=int(np.argmin(np.linalg.norm(route-env.ego_state[:2],axis=1)))
            item=dict(step=step,active_ids=np.flatnonzero(env.agent_active).tolist(),remaining_route_m=float(arc[-1]-arc[nearest]),
                current_CV=area(reach,env,False),no_traffic=area(reach,env,False,empty=True))
            if joint is not None:item['recorded_executed_candidate']=area(reach,env,True)
            else:
                # 本步新批次在解码前拒绝；只评估上一条完整已记录未来的剩余部分，不虚构原始7样本。
                prev=next(r['joint'] for r in reversed(rows) if r['kind']=='execution_prediction' and r['step']<step and r.get('joint'))
                prev=decode(prev);offset=step-prev.source_step
                cached=replace(prev,source_step=step,positions_global=prev.positions_global[:,offset:],
                    velocities_global=prev.velocities_global[:,offset:],yaws_global=prev.yaws_global[:,offset:],valid_mask=prev.valid_mask[:,offset:])
                env.pending_joint_trajectory=cached
                item['previous_joint_tail']=dict(original_source_step=prev.source_step,remaining_steps=cached.positions_global.shape[1],
                    all_current_active_covered=set(np.flatnonzero(env.agent_active))==set(cached.agent_ids))
                gate=JointTrajectorySafetyGate(cfg.sim)
                try:
                    corrected=project_joint(cached,env.data_dict['agent'][-1],.1,gate.dynamic_limits,cfg.sim.traffic_model.dynamics_projection,previous)
                    env.pending_joint_trajectory=corrected;gate.validate_joint(env)
                    validate_execution_prefix(env.data_dict['agent'][-1],corrected,env.get_scenario_frame().static_obstacles_global)
                    item['previous_joint_tail'].update(physical_and_background_pass=True,theoretical_area=area(reach,env,True))
                except Exception as error:item['previous_joint_tail'].update(physical_and_background_pass=False,error_type=type(error).__name__,reason=str(error))
            results.append(item)
            print('FRAME',index,step,json.dumps(item,ensure_ascii=False),flush=True)
        records.append(dict(scenario_index=index,reason=a['generation_failure_reason'],states=results,
            source_trace_sha256=hashlib.sha256(p.with_name('execution_trace.jsonl').read_bytes()).hexdigest(),
            source_scene_sha256=hashlib.sha256(Path(a['scenario_source']).read_bytes()).hexdigest()))
    result=dict(records=records,real_LLM_requests=0,diffusion_samples=0,actual_executed_frames=0,
        scope='offline same-state geometry/dynamics replay; no closed-loop recovery or majority-coverage claim')
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
