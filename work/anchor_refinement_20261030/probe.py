"""真实预训练扩散固定计划配对；CPU无API，原8样本/求解器/权重及安全不变。"""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,copy,hashlib,json,os,random,sys,time
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from objective import MODES,install_anchor_objective


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--modes',choices=MODES,nargs='+',default=[MODES[0]])
    a=p.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_probe_requires_GPU_hidden')
    if a.output.exists():raise FileExistsError('preserve_prior_probe')
    torch.set_num_threads(2)
    for k in ['PROJECT_ROOT','SCRATCH_ROOT']:os.environ.setdefault(k,str(ROOT))
    os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'))
    from simulator import Simulator
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    from policies.diffusion_model_wrapper import DiffusionModelWrapper
    from policies.trajectory_projection import project_positions
    from policies.ego_profile import JointTrajectorySafetyGate
    from policies.joint_safety import NoSafeJointCandidate
    from policies.risk_metrics import RouteReachability
    source=ROOT/'experiments/skills_temporal_candidates_scene074_20261029_r1'
    trace=next((source/'temporal_full').rglob('execution_trace.jsonl'))
    rows=[json.loads(s) for s in trace.read_text().splitlines()]
    plan=next(r['validated_plan'] for r in rows if r.get('kind')=='llm_output' and r['step']==0)
    frame=next(r for r in rows if r.get('kind')=='state' and r['step']==0)
    manifest=json.loads((source/'manifest.json').read_text())
    cfg=OmegaConf.load(source/'temporal_full/resolved_config.yaml')
    cfg.sim.traffic_model.device='cpu';cfg.sim.traffic_model.multi_gpu_devices=[]
    cfg.sim.visualize=False;cfg.sim.save_carla_data=False
    files=[ROOT/n for n in json.loads((source/'source_sha256.json').read_text())]
    files += list(HERE.glob('*.py'))
    before={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    a.output.mkdir();(a.output/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    save(a.output/'launch.json',dict(modes=a.modes,seeds=a.seeds,source_trace=str(trace),source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        plan=plan,source_sha256=before,planned_predictions=len(a.modes)*len(a.seeds),device='CPU',new_API_requests=0,new_execution_frames=0,
        scope='fixed_plan_diffusion_generation_probe_not_closed_loop_or_training',started_at=time.time()))
    def forbidden(*args,**kwargs):raise AssertionError('API_forbidden_in_fixed_plan_diffusion_probe')
    LLMAdversarialPlanner._request_attack_plan=forbidden
    env=Simulator(cfg);env.test_files=manifest['scenario_files'];records=[]
    original=DiffusionModelWrapper._select_joint_guided_action
    seed=mode=None
    def capture(wrapper,net,batch,action,info,anchor_tensors=None):
        selected,index=original(wrapper,net,batch,action,info,anchor_tensors)
        target=anchor_tensors['llm_anchor_positions_local'].to(selected.positions.device)
        mask=anchor_tensors['llm_anchor_mask'].to(selected.positions.device)
        events=torch.zeros_like(mask);events[:,[9,19,29]]=mask[:,[9,19,29]]
        errors=torch.linalg.norm(selected.positions-target,dim=-1)
        active_rows=torch.nonzero(mask.any(1)).flatten();row=int(active_rows[0])
        data=net._prepare_guidance_data(batch);transform=data['world_from_agent'][row].detach().cpu().numpy()
        current=np.asarray(env.data_dict['agent'][-1][plan['attack_target_id']],float)
        positions=selected.positions[row].detach().cpu().numpy()@transform[:2,:2].T+transform[:2,2]
        r=dict(seed=seed,mode=mode,raw_dense_mean_error_m=float(errors[mask].mean()),
            raw_anchor_mean_error_m=float(errors[events].mean()),raw_errors_1_2_3s_m=errors[row,[9,19,29]].detach().cpu().tolist(),
            actual_loss_weights=net.Loss_Calculater.weights.detach().cpu().tolist(),selected_sample=index,projection=None)
        projected=np.empty((0,2));limits=JointTrajectorySafetyGate(cfg.sim).dynamic_limits
        try:
            setting=cfg.sim.traffic_model.dynamics_projection
            projected,velocity,yaw,details=project_positions(positions,current,.1,limits,float(setting.max_seconds_per_vehicle),float(setting.max_deviation_m))
            local=(projected-transform[:2,2])@transform[:2,:2];delta=np.linalg.norm(local-target[row].detach().cpu().numpy(),axis=1)
            joint=NS(agent_ids=np.array([plan['attack_target_id']]),positions_global=projected[None],velocities_global=velocity[None],yaws_global=yaw[None],valid_mask=np.ones((1,len(projected)),bool))
            replay=NS(data_dict=env.data_dict,agent_active=env.agent_active,ego_state=env.ego_state,scenario_dict=env.scenario_dict,
                pending_joint_trajectory=joint,dt=.1,previously_attacked_agent_ids=(),get_static_obstacles=env.get_static_obstacles)
            JointTrajectorySafetyGate(cfg.sim).validate_joint(replay)
            reach=RouteReachability(cfg.sim.evaluation.reachability)
            baseline=reach.compute(replay,False,[]);danger=reach.compute(replay,True,[])
            r['projection']=dict(dense_mean_error_m=float(delta[mask[row].cpu().numpy()].mean()),anchor_errors_1_2_3s_m=delta[[9,19,29]].tolist(),
                anchor_mean_error_m=float(delta[[9,19,29]].mean()),details=details,existing_joint_gate_pass=True,
                baseline_area_m2=baseline['area_m2'],dangerous_area_m2=danger['area_m2'],positive_area=baseline['area_m2']>0 and danger['area_m2']>0)
        except NoSafeJointCandidate as exc:r['projection_rejection']=str(exc)
        np.savez_compressed(a.output/f'{mode}_seed{seed}.npz',samples_local=info['action_samples']['positions'].detach().cpu().numpy(),
            selected_local=selected.positions.detach().cpu().numpy(),target_local=target.detach().cpu().numpy(),mask=mask.cpu().numpy(),
            projected_global=projected,world_from_agent=transform,source_state=current)
        records.append(r);save(a.output/'results.partial.json',dict(records=records,new_API_requests=0,new_execution_frames=0))
        return selected,index
    DiffusionModelWrapper._select_joint_guided_action=capture
    try:
        for seed in a.seeds:
            for mode in a.modes:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);env.reset(0)
                assert np.array_equal(np.asarray(env.ego_state),np.asarray(frame['ego']))
                assert np.allclose(env.data_dict['agent'][-1][env.agent_active],np.asarray(frame['agents'])[np.asarray(frame['active'])],atol=1e-6)
                assert int(np.count_nonzero(env.agent_active))==1
                params=dict(cfg.sim.traffic_model.guidance.params)
                assert params['inner_lr']==.2 and params['inner_beta']==.5 and params['n_guide_steps']==2
                weights=list(cfg.sim.traffic_model.guidance.llm_joint.weights)
                assert env.diffusion_controller.update_iterative_guidance(params,weights)
                env.set_attack_intent(target_id=plan['attack_target_id'],anchors=plan['anchors'],strategy=plan['strategy'])
                env.attack_intent.update(duration=plan.get('duration',10),request_id=plan.get('request_id'))
                # 重置后再次固定采样RNG，避免模型初始化或缓存影响两种目标的同噪声比较。
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                with install_anchor_objective(mode):env.prepare_background_traffic()
    finally:DiffusionModelWrapper._select_joint_guided_action=original
    intact=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in before.items())
    save(a.output/'results.json',dict(records=records,source_unchanged=intact,new_API_requests=0,new_execution_frames=0,goal_complete=False))
    save(a.output/'exit_status.json',dict(exit_code=0 if intact else 1,finished_at=time.time()))
    print(json.dumps(dict(predictions=len(records),source_unchanged=intact)))


if __name__=='__main__':
    try:main()
    except Exception as error:
        # 参数错误/运行异常保留原日志；不自动换种子、放宽约束或重试。
        print(json.dumps(dict(failure_type=type(error).__name__)),flush=True);raise
