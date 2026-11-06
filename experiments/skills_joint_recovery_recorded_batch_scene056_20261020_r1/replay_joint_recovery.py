"""在完全相同的已冻结批次上验收回退，不重新采样或更改任何安全参数。"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
from policies.traffic_types import JointTrajectory
from policies.ego_profile import JointTrajectorySafetyGate
from policies.trajectory_projection import project_joint
from policies.joint_safety import NoSafeJointCandidate
from joint_candidate_recovery import recover_joint

OUT=ROOT/'experiments/skills_joint_recovery_recorded_batch_scene056_20261020_r1'
BASE=ROOT/'experiments/skills_v2_frozen_development10_cpu_pair_20261020_r1/skills_v2'
FIXTURE=ROOT/'experiments/skills_joint_batch_feasibility_scene056_cpu_20261020_r2'


def main():
    OUT.mkdir(exist_ok=False)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    cfg=OmegaConf.load(BASE/'resolved_config.yaml');settings=cfg.sim.traffic_model.dynamics_projection
    trace=BASE/'movies/scenario_056/execution_trace.jsonl';rows=[json.loads(l) for l in trace.open()]
    state=next(x for x in rows if x['kind']=='state' and x['step']==0)
    context=next(x['state'] for x in rows if x['kind']=='llm_input' and x['step']==0)
    raw=next((FIXTURE/'samples').glob('*.npz'));batch=np.load(raw)
    samples=batch['positions'];states=np.asarray(state['agents'],float);ids=batch['agent_ids']
    selected=int(batch['selected_index']);candidates=[]
    for index in range(1,len(samples)):
        velocity=np.diff(np.concatenate((states[ids,None,:2],samples[index]),axis=1),axis=1)/.1
        yaw=np.arctan2(velocity[:,:,1],velocity[:,:,0])[...,None]
        candidates.append(JointTrajectory(source_step=0,agent_ids=ids,positions_global=samples[index],
            velocities_global=velocity,yaws_global=yaw,valid_mask=np.ones(samples[index].shape[:2],bool),
            metadata={'joint_sample_index':index}))
    native=next(c for c in candidates if c.metadata['joint_sample_index']==selected)
    env=NS(cfg=cfg,data_dict={'agent':[states]},agent_active=np.asarray(state['active'],bool),
        ego_state=np.asarray(state['ego']),scenario_dict={'route':context['route']},dt=.1,
        get_static_obstacles=lambda:state['static_obstacles'],previously_attacked_agent_ids=(),
        get_scenario_frame=lambda:NS(static_obstacles_global=np.empty((0,5))))
    limits=JointTrajectorySafetyGate(cfg.sim).dynamic_limits;results={}
    for enabled in (False,True):
        started=time.monotonic()
        try:
            output=recover_joint(env,native,candidates,project_joint,states,.1,limits,settings,enabled=enabled)
            results[str(enabled)]=dict(passed=True,metadata=output.metadata)
            np.savez_compressed(OUT/'recovered_prediction.npz',positions_global=output.positions_global,
                                raw_positions_global=samples[output.metadata['joint_sample_index']],
                                initial_states=states,agent_ids=ids)
        except NoSafeJointCandidate as error:results[str(enabled)]=dict(passed=False,reason=str(error))
        results[str(enabled)]['elapsed_seconds']=time.monotonic()-started
    assert results['False']['reason']=='projection_excessive_deviation'
    assert results['True']['passed'] and results['True']['metadata']['joint_sample_index']==4
    record=dict(results=results,limits=limits,max_deviation_m=float(settings.max_deviation_m),
        max_seconds_per_vehicle=float(settings.max_seconds_per_vehicle),
        input_batch_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),LLM_calls=0,actual_executed_frames=0,
        scope='identical recorded original batch, original gates; no closed-loop success claim')
    (OUT/'probe_result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
    (OUT/'exit_status.json').write_text(json.dumps(dict(exit_code=0)))


if __name__=='__main__':main()
