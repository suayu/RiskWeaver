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
    p.add_argument('--scene',type=int,choices=[74,182,56],required=True)
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
    names={74:('skills_temporal_candidates_scene074_20261029_r1','temporal_full'),182:('skills_priority_v5_closed_loop_scene182_20261025_r1','priority_v5'),56:('skills_modelpool_recovery_development10_paired_20261023_r3','skills_full_shared_recovery')}
    name,arm=names[a.scene];source=ROOT/'experiments'/name
    trace=source/arm/'movies'/f'scenario_{a.scene:03d}'/'execution_trace.jsonl'
    rows=[json.loads(s) for s in trace.read_text().splitlines()]
    plan=next(r['validated_plan'] for r in rows if r.get('kind')=='llm_output' and r['step']==0)
    frame=next(r for r in rows if r.get('kind')=='state' and r['step']==0)
    manifest=json.loads((source/'manifest.json').read_text())
    cfg=OmegaConf.load(source/arm/'resolved_config.yaml')
    manifest_index=manifest['original_scenario_indices'].index(a.scene)
    selected_file=manifest['scenario_files'][manifest_index]
    if hashlib.sha256(Path(selected_file).read_bytes()).hexdigest()!=manifest['scenario_sha256'][manifest_index]:raise ValueError('frozen_scene_hash_mismatch')
    cfg.sim.traffic_model.device='cpu';cfg.sim.traffic_model.multi_gpu_devices=[]
    cfg.sim.visualize=False;cfg.sim.save_carla_data=False
    cfg.sim.scenario_data_output_path=str(a.output/'diagnostic_initial_data')
    cfg.sim.movie_path=str(a.output/'unused_movies')
    files=[ROOT/n for n in json.loads((source/'source_sha256.json').read_text())]
    files += list(HERE.glob('*.py'))
    before={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    a.output.mkdir();(a.output/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    save(a.output/'launch.json',dict(modes=a.modes,seeds=a.seeds,source_trace=str(trace),source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        scene=a.scene,scene_file=selected_file,plan=plan,source_sha256=before,planned_predictions=len(a.modes)*len(a.seeds),device='CPU',new_API_requests=0,new_execution_frames=0,
        scope='fixed_plan_diffusion_generation_probe_not_closed_loop_or_training',started_at=time.time()))
    def forbidden(*args,**kwargs):raise AssertionError('API_forbidden_in_fixed_plan_diffusion_probe')
    LLMAdversarialPlanner._request_attack_plan=forbidden
    env=Simulator(cfg);env.test_files=[selected_file];records=[]
    original=DiffusionModelWrapper._select_joint_guided_action
    seed=mode=None
    def capture(wrapper,net,batch,action,info,anchor_tensors=None):
        selected,index=original(wrapper,net,batch,action,info,anchor_tensors)
        target=anchor_tensors['llm_anchor_positions_local'].to(selected.positions.device)
        mask=anchor_tensors['llm_anchor_mask'].to(selected.positions.device)
        events=torch.zeros_like(mask);events[:,[9,19,29]]=mask[:,[9,19,29]]
        errors=torch.linalg.norm(selected.positions-target,dim=-1)
        active_rows=torch.nonzero(mask.any(1)).flatten();row=int(active_rows[0])
        data=net._prepare_guidance_data(batch);transforms=data['world_from_agent'].detach().cpu().numpy();transform=transforms[row]
        active_ids=np.flatnonzero(env.agent_active)
        current_agents=np.asarray(env.data_dict['agent'][-1],float)
        distances=np.linalg.norm(transforms[:,None,:2,2]-current_agents[None,active_ids,:2],axis=-1)
        agent_ids=active_ids[np.argmin(distances,axis=1)]
        if np.max(np.min(distances,axis=1))>1e-4 or len(set(agent_ids.tolist()))!=len(agent_ids):raise ValueError('joint_agent_mapping_not_unique')
        assert agent_ids[row]==plan['attack_target_id']
        current=np.asarray(env.data_dict['agent'][-1][plan['attack_target_id']],float)
        positions=selected.positions[row].detach().cpu().numpy()@transform[:2,:2].T+transform[:2,2]
        r=dict(seed=seed,mode=mode,raw_dense_mean_error_m=float(errors[mask].mean()),
            raw_anchor_mean_error_m=float(errors[events].mean()),raw_errors_1_2_3s_m=errors[row,[9,19,29]].detach().cpu().tolist(),
            actual_loss_weights=net.Loss_Calculater.weights.detach().cpu().tolist(),selected_sample=index,projection=None)
        r['proximal_audit']=next((m.anchor_proximal_audit for m in net.modules() if hasattr(m,'anchor_proximal_audit')),None) if mode=='dense_squared_proximal' else None
        projected=np.empty((0,2));limits=JointTrajectorySafetyGate(cfg.sim).dynamic_limits
        try:
            setting=cfg.sim.traffic_model.dynamics_projection
            all_positions=selected.positions.detach().cpu().numpy()
            all_positions=np.einsum('bti,bji->btj',all_positions,transforms[:,:2,:2])+transforms[:,None,:2,2]
            projected_rows=[];velocities=[];yaws=[];projection_details=[]
            for j,agent_id in enumerate(agent_ids):
                pp,vv,yy,dd=project_positions(all_positions[j],current_agents[agent_id],.1,limits,float(setting.max_seconds_per_vehicle),float(setting.max_deviation_m))
                projected_rows.append(pp);velocities.append(vv);yaws.append(yy);projection_details.append(dd)
            projected=np.asarray(projected_rows)[row];details=projection_details[row]
            local=(projected-transform[:2,2])@transform[:2,:2];delta=np.linalg.norm(local-target[row].detach().cpu().numpy(),axis=1)
            joint=NS(agent_ids=agent_ids,positions_global=np.asarray(projected_rows),velocities_global=np.asarray(velocities),yaws_global=np.asarray(yaws),valid_mask=np.ones((len(agent_ids),len(projected)),bool))
            replay=NS(data_dict=env.data_dict,agent_active=env.agent_active,ego_state=env.ego_state,scenario_dict=env.scenario_dict,
                pending_joint_trajectory=joint,dt=.1,previously_attacked_agent_ids=(),get_static_obstacles=env.get_static_obstacles)
            JointTrajectorySafetyGate(cfg.sim).validate_joint(replay)
            reach=RouteReachability(cfg.sim.evaluation.reachability)
            baseline=reach.compute(replay,False,[]);danger=reach.compute(replay,True,[])
            r['projection']=dict(dense_mean_error_m=float(delta[mask[row].cpu().numpy()].mean()),anchor_errors_1_2_3s_m=delta[[9,19,29]].tolist(),
                anchor_mean_error_m=float(delta[[9,19,29]].mean()),details=details,existing_joint_gate_pass=True,
                baseline_area_m2=baseline['area_m2'],dangerous_area_m2=danger['area_m2'],positive_area=baseline['area_m2']>0 and danger['area_m2']>0)
            r['projection']['all_agent_details']=projection_details
        except NoSafeJointCandidate as exc:r['projection_rejection']=str(exc)
        np.savez_compressed(a.output/f'{mode}_seed{seed}.npz',samples_local=info['action_samples']['positions'].detach().cpu().numpy(),
            selected_local=selected.positions.detach().cpu().numpy(),target_local=target.detach().cpu().numpy(),mask=mask.cpu().numpy(),
            projected_global=projected,world_from_agent=transform,source_state=current,agent_ids=agent_ids)
        records.append(r);save(a.output/'results.partial.json',dict(records=records,new_API_requests=0,new_execution_frames=0))
        return selected,index
    DiffusionModelWrapper._select_joint_guided_action=capture
    try:
        for seed in a.seeds:
            for mode in a.modes:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);env.reset(0)
                assert np.array_equal(np.asarray(env.ego_state),np.asarray(frame['ego']))
                assert np.allclose(env.data_dict['agent'][-1][env.agent_active],np.asarray(frame['agents'])[np.asarray(frame['active'])],atol=1e-6)
                params=dict(cfg.sim.traffic_model.guidance.params)
                assert params['inner_lr']==.2 and params['inner_beta']==.5 and params['n_guide_steps']==2
                weights=list(cfg.sim.traffic_model.guidance.llm_joint.weights)
                assert env.diffusion_controller.update_iterative_guidance(params,weights)
                env.set_attack_intent(target_id=plan['attack_target_id'],anchors=plan['anchors'],strategy=plan['strategy'])
                env.attack_intent.update(duration=plan.get('duration',10),request_id=plan.get('request_id'))
                # 重置后再次固定采样RNG，避免模型初始化或缓存影响两种目标的同噪声比较。
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                count_before=len(records)
                started=time.monotonic()
                try:
                    with install_anchor_objective(mode):env.prepare_background_traffic()
                except NoSafeJointCandidate as exc:
                    if len(records)==count_before:records.append(dict(seed=seed,mode=mode,prediction_rejection=str(exc)))
                    else:records[-1]['downstream_existing_gate_rejection']=str(exc)
                    save(a.output/'results.partial.json',dict(records=records,new_API_requests=0,new_execution_frames=0))
                if len(records)>count_before:
                    records[-1]['prediction_and_validation_seconds']=time.monotonic()-started
                    save(a.output/'results.partial.json',dict(records=records,new_API_requests=0,new_execution_frames=0))
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
