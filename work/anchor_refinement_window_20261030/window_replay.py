"""固定74号已接收计划的10帧执行诊断；零LLM，仅检验细化与投影后的跟随。"""
from pathlib import Path
import argparse, hashlib, json, os, random, sys, time
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from objective import install_anchor_objective


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['legacy_dense_huber','dense_squared_proximal'],required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only')
    if a.output.exists():raise FileExistsError('preserve_prior_run')
    torch.set_num_threads(2)
    for k in ['PROJECT_ROOT','SCRATCH_ROOT']:os.environ.setdefault(k,str(ROOT))
    os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'))
    from simulator import Simulator
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from policies.idm_policy import IDMPolicy
    from policies.evaluation_trace import AttemptTrace
    from policies.trajectory_projection import project_joint
    from policies.ego_profile import JointTrajectorySafetyGate
    from policies.joint_safety import NoSafeJointCandidate,validate_execution_prefix
    from policies.risk_metrics import RouteReachability
    from experiments.comparison.e1_metrics import ExecutionMetrics
    sys.path.insert(0,str(ROOT/'work/skills_video_export_fix_20261023'))
    from integer_frame_export import encode
    source=ROOT/'experiments/skills_temporal_candidates_scene074_20261029_r1'
    original_trace=source/'temporal_full/movies/scenario_074/execution_trace.jsonl'
    rows=[json.loads(x) for x in original_trace.read_text().splitlines()]
    plan=next(x['validated_plan'] for x in rows if x.get('kind')=='llm_output' and x['step']==0)
    state=next(x for x in rows if x.get('kind')=='state' and x['step']==0)
    manifest=json.loads((source/'manifest.json').read_text())
    cfg=OmegaConf.load(source/'temporal_full/resolved_config.yaml')
    cfg.sim.traffic_model.device='cpu';cfg.sim.traffic_model.multi_gpu_devices=[]
    cfg.sim.visualize=True;cfg.sim.save_carla_data=False
    cfg.sim.scenario_data_output_path=str(a.output/'diagnostic_data');cfg.sim.movie_path=str(a.output/'frames')
    before={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in json.loads((source/'source_sha256.json').read_text())}
    for file in HERE.glob('*.py'):before[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    a.output.mkdir();dump(a.output/'source_sha256.json',before)
    dump(a.output/'launch.json',dict(mode=a.mode,scene=74,seed=42,planned_frames=10,source_plan=plan,
        source_trace_sha256=hashlib.sha256(original_trace.read_bytes()).hexdigest(),API_calls=0,
        scope='one_logged_fixed_plan_ten_frame_tracking_diagnostic_not_new_planner_success_or_full_episode',started_at=time.time()))
    (a.output/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding='utf-8')
    def forbidden(*args,**kwargs):raise AssertionError('LLM_API_forbidden')
    LLMAdversarialPlanner._request_attack_plan=forbidden
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    env=Simulator(cfg);env.test_files=manifest['scenario_files'];obs=env.reset(0)
    if not np.array_equal(np.asarray(env.ego_state),np.asarray(state['ego'])):raise ValueError('ego_initial_mismatch')
    if not np.allclose(env.data_dict['agent'][-1][env.agent_active],np.asarray(state['agents'])[np.asarray(state['active'])],atol=1e-6):raise ValueError('traffic_initial_mismatch')
    policy=IDMPolicy(cfg,env);policy.reset(obs);out=a.output/'frames';out.mkdir()
    trace=AttemptTrace(out);trace.state(env,initial=True)
    env.set_attack_intent(target_id=plan['attack_target_id'],anchors=plan['anchors'],strategy=plan['strategy'])
    env.attack_intent.update(duration=10,request_id='fixed-plan-anchor-refinement-diagnostic')
    controller=env.diffusion_controller
    if not controller.update_iterative_guidance(dict(cfg.sim.traffic_model.guidance.params),list(cfg.sim.traffic_model.guidance.llm_joint.weights)):raise ValueError('guidance_not_applied')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    gate=JointTrajectorySafetyGate(cfg.sim);reach=RouteReachability(cfg.sim.evaluation.reachability)
    execution_metrics=ExecutionMetrics(env)
    records=[];failure=None;terminated=False;started=time.monotonic()
    with install_anchor_objective(a.mode):
        for step in range(10):
            env.render_state(name='scenario_074',movie_path=str(out))
            try:
                joint=env.prepare_background_traffic()
                trace.append(dict(kind='proposal_prediction',step=step,positions_global=joint.positions_global.tolist(),metadata=joint.metadata))
                corrected=project_joint(joint,np.asarray(env.data_dict['agent'][-1],float),env.dt,gate.dynamic_limits,cfg.sim.traffic_model.dynamics_projection,getattr(env,'previously_attacked_agent_ids',()))
                env.inject_joint_trajectory(corrected);gate.validate_joint(env)
                validate_execution_prefix(env.data_dict['agent'][-1],corrected,env.get_scenario_frame().static_obstacles_global)
                baseline=reach.compute(env,False,[]);danger=reach.compute(env,True,[])
                if baseline['area_m2']<=0 or danger['area_m2']<=0:raise NoSafeJointCandidate('zero_theoretical_area')
                trace.append(dict(kind='execution_prediction',step=step,positions_global=corrected.positions_global.tolist(),metadata=corrected.metadata))
                obs,terminated,info=env.step(policy.act(obs),np.asarray(plan['anchors']),env.get_attack_target_prediction())
                execution_metrics.observe(env,info)
                trace.state(env,info);execution=trace.attack_execution(env,corrected)
                t=(step+1)*env.dt;anchors=np.asarray(plan['anchors'])
                target=np.array([np.interp(t,np.arange(len(anchors)),anchors[:,axis]) for axis in (0,1)])
                actual=np.asarray(env.data_dict['agent'][-1][plan['attack_target_id']],float)
                records.append(dict(step=step+1,time_s=t,actual_target=actual.tolist(),ego=np.asarray(env.ego_state).tolist(),
                    interpolated_anchor=target.tolist(),tracking_error_m=float(np.linalg.norm(actual[:2]-target)),
                    reachable_area_m2=danger['area_m2'],execution=execution,info=info))
                if terminated:break
            except NoSafeJointCandidate as error:
                failure=str(error);trace.append(dict(kind='generation_failure',step=step,reason=failure));break
    env.render_state(name='scenario_074',movie_path=str(out))
    video=encode(out,out/'execution_trace.jsonl',a.output.with_name(a.output.name+'_video')/'scenario_074.mp4',a.output,fps=5)
    intact=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in before.items())
    result=dict(mode=a.mode,planned_frames=10,executed_frames=len(records),attack_executed_frames=trace.attack_executed_frames,
        generation_failure_reason=failure,terminated=terminated,records=records,video=video,API_calls=0,
        runtime_seconds=time.monotonic()-started,source_unchanged=intact,D=None,D_not_measured=True,
        full_episode=False,execution_metrics=execution_metrics.result(),scope='fixed_plan_short_execution_tracking_not_planner_or_collision_rate_evaluation')
    dump(a.output/'attempt_result.json',result);dump(a.output/'exit_status.json',dict(exit_code=0 if intact else 1))
    print(json.dumps(dict(executed_frames=len(records),failure=failure,source_unchanged=intact)))


if __name__=='__main__':main()
