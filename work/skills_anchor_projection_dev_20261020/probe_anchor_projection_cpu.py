"""三组预先固定随机样本的真实权重单帧候选诊断；零LLM请求、零实际执行。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import random
from types import SimpleNamespace as NS
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
# 与共用E1入口一致，只设置路径变量，不读取或复制密钥文件。
os.environ.setdefault('PROJECT_ROOT',str(ROOT))
os.environ.setdefault('SCRATCH_ROOT',str(ROOT))
os.environ.setdefault('DATASET_ROOT',str(ROOT/'metadata'))
from experiments.comparison.e1_protocol import group_overrides
from policies.diffusion_model_wrapper import DiffusionModelWrapper
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from simulator import Simulator
sys.path.insert(0,str(ROOT/'work/skills_v2_compact_dev_20261020'))
from single_actor_anchor_selection import select_single_actor
from policies.trajectory_projection import project_positions
from policies.ego_profile import JointTrajectorySafetyGate
from policies.joint_safety import NoSafeJointCandidate
from policies.risk_metrics import RouteReachability

OUT=ROOT/'experiments/skills_anchor_projection_cpu_probe_scene182_20261020_r1'
SEEDS=[182000588,182000589,182000590]
WEIGHTS=[16.,32.,64.]


def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU_probe_requires_no_visible_GPU')
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    source=ROOT/'experiments/skills_stage_d_paired_scene182_bailian_20261020_r1/skills_expanded/movies/scenario_182/execution_trace.jsonl'
    rows=[json.loads(line) for line in source.open()]
    plan=next(r['validated_plan'] for r in rows if r['kind']=='llm_output' and r['step']==0)
    if not plan or not plan.get('attack'):raise RuntimeError('fixed_probe_plan_not_available')
    manifest=json.loads((source.parents[3]/'manifest.json').read_text())
    # source.parents[3]为根冒烟目录；固定输入和计划，不搜索更有利的场景。
    dump('launch.json',dict(seeds=SEEDS,anchor_weights=WEIGHTS,device='cpu',solver_fixed=dict(inner_lr=.2,inner_beta=.5,n_guide_steps=2),scene=182,plan=plan,source_trace=str(source),
        source_trace_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),LLM_calls=0,actual_executed_frames=0,
        scope='CPU weight and existing-projection response probe; zero anchor weight still retains TTC/collision guidance; same three seeds per weight; no speed comparison to GPU; no execution',started_at=time.time()))
    def forbidden(*args,**kwargs):raise AssertionError('LLM_API_forbidden_in_native_candidate_probe')
    LLMAdversarialPlanner._request_attack_plan=forbidden
    overrides=group_overrides('riskweaver_profile_off','idm',42,400,'qwen3.8-flash',['qwen3.8-flash'])
    overrides=['sim.visualize=false' if x=='sim.visualize=true' else x for x in overrides]+['sim.traffic_model.device=cpu']
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'cfgs')):
        cfg=compose(config_name='config',overrides=overrides)
    (OUT/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding='utf8')
    env=Simulator(cfg);env.test_files=manifest['scenario_files']
    original=DiffusionModelWrapper._select_joint_guided_action
    records=[];current_seed=None;current_weight=None
    def compare(wrapper,policy_net,model_batch,action,info,anchor_tensors=None):
        native,native_index=original(wrapper,policy_net,model_batch,action,info,anchor_tensors)
        native_audit=copy.deepcopy(getattr(wrapper,'_last_anchor_audit',None))
        proposed,index=select_single_actor(wrapper,original,policy_net,model_batch,action,info,anchor_tensors)
        audit=copy.deepcopy(getattr(wrapper,'_last_anchor_audit',None))
        samples=info['action_samples'];positions=samples['positions'];targets=anchor_tensors['llm_anchor_positions_local'].to(positions.device)
        mask=anchor_tensors['llm_anchor_mask'].to(positions.device)
        def metrics(candidate):
            errors=torch.linalg.norm(candidate.positions-targets,dim=-1)[mask]
            return dict(mean_error_m=float(errors.mean()),max_error_m=float(errors.max()))
        native_metrics=metrics(native);proposed_metrics=metrics(proposed)
        record=dict(seed=current_seed,anchor_weight=current_weight,actual_loss_weights=policy_net.Loss_Calculater.weights.detach().cpu().tolist(),actors=int(positions.shape[0]),samples=int(positions.shape[1]),
            native_index=native_index,proposed_index=index,native=native_metrics,proposed=proposed_metrics,
            native_anchor_audit=native_audit,proposed_anchor_audit=audit,
            anchor_score_weight=wrapper.anchor_candidate_score_weight,
            stage='before downstream dynamics projection and execution gates; no action executed')
        records.append(record)
        data=policy_net._prepare_guidance_data(model_batch)
        transform=data['world_from_agent'][0].detach().cpu().numpy()
        if positions.shape[0]!=1 or np.count_nonzero(env.agent_active)!=1:
            raise RuntimeError('probe requires exactly one active background actor')
        initial=np.asarray(env.data_dict['agent'][-1][plan['attack_target_id']],float)
        record['input_state_sha256']=hashlib.sha256(np.r_[env.ego_state,initial].tobytes()).hexdigest()
        global_native=native.positions[0].detach().cpu().numpy()@transform[:2,:2].T+transform[:2,2]
        projected=np.empty((0,2))
        limits=JointTrajectorySafetyGate(cfg.sim).dynamic_limits
        settings=cfg.sim.traffic_model.dynamics_projection
        try:
            projected,velocity,yaw,projection=project_positions(global_native,initial,.1,limits,
                 float(settings.max_seconds_per_vehicle),float(settings.max_deviation_m))
            local_projected=(projected-transform[:2,2])@transform[:2,:2]
            delta=np.linalg.norm(local_projected-targets[0].detach().cpu().numpy(),axis=1)
            record['projected']=dict(mean_error_m=float(delta[mask[0].cpu().numpy()].mean()),
                 max_error_m=float(delta[mask[0].cpu().numpy()].max()),endpoint_3s_error_m=float(delta[29]),
                 projection=projection)
            # 与执行门控使用相同状态、动力学和几何；此处不会调用env.step。
            joint=NS(agent_ids=np.asarray([plan['attack_target_id']]),positions_global=projected[None],
                     velocities_global=velocity[None],yaws_global=yaw[None],valid_mask=np.ones((1,len(projected)),bool))
            replay=NS(data_dict=env.data_dict,agent_active=env.agent_active,ego_state=env.ego_state,
                      scenario_dict=env.scenario_dict,pending_joint_trajectory=joint,dt=.1,
                      previously_attacked_agent_ids=(),get_static_obstacles=env.get_static_obstacles)
            JointTrajectorySafetyGate(cfg.sim).validate_joint(replay)
            reach=RouteReachability(cfg.sim.evaluation.reachability)
            baseline=reach.compute(replay,False,[]);danger=reach.compute(replay,True,[])
            record['projected'].update(existing_joint_gate_pass=True,original_area_m2=baseline['area_m2'],
                 dangerous_area_m2=danger['area_m2'],positive_area_gate_pass=bool(baseline['area_m2']>0 and danger['area_m2']>0))
        except NoSafeJointCandidate as error:
            record['projection_or_gate_rejection']=str(error)
        record['original_limits']=limits

        np.savez_compressed(OUT/f'candidates_w{current_weight:g}_seed_{current_seed}.npz',
            positions_local=positions.detach().cpu().numpy(),
            targets_local=targets.detach().cpu().numpy(),mask=mask.detach().cpu().numpy(),
            native_local=native.positions.detach().cpu().numpy(),proposed_local=proposed.positions.detach().cpu().numpy(),projected_global=projected,
            world_from_agent=transform,anchors_global=np.asarray(plan['anchors']),dt=.1)
        wrapper._last_anchor_audit=native_audit
        # 本诊断仍返回原动作；修正动作只作为反事实数据，不进入仿真执行。
        return native,native_index
    DiffusionModelWrapper._select_joint_guided_action=compare
    try:
        assert env.diffusion_controller.device.type=='cpu'
        for weight in WEIGHTS:
            for seed in SEEDS:
                current_seed=seed;current_weight=weight
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                cfg.sim.scenario_data_output_path=str(OUT/f'debug_w{weight:g}_seed_{seed}')
                env.reset(0)
                fixed=dict(cfg.sim.traffic_model.guidance.params)
                expected=[1.5,1.,2.,weight]
                assert env.diffusion_controller.update_iterative_guidance(fixed,expected)
                net=env.diffusion_controller.policy.nets['policy']
                assert np.allclose(net.Loss_Calculater.weights.detach().cpu().tolist(),expected)
                assert fixed['inner_lr']==.2 and fixed['inner_beta']==.5 and fixed['n_guide_steps']==2
                env.set_attack_intent(target_id=plan['attack_target_id'],anchors=plan['anchors'],strategy=plan['strategy'])
                env.attack_intent.update(duration=plan.get('duration',10),request_id=plan.get('request_id'))
                env.prepare_background_traffic()
                dump('probe_result.partial.json',dict(records=records,LLM_calls=0,actual_executed_frames=0))
    finally:DiffusionModelWrapper._select_joint_guided_action=original
    assert len(records)==len(SEEDS)*len(WEIGHTS)
    assert all(len({r['input_state_sha256'] for r in records if r['seed']==s})==1 for s in SEEDS)
    dump('probe_result.json',dict(records=records,LLM_calls=0,actual_executed_frames=0,finished_at=time.time()))
    dump('exit_status.json',dict(exit_code=0,finished_at=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():dump('exit_status.json',dict(exit_code=1,error_type=type(error).__name__,finished_at=time.time()))
        raise
