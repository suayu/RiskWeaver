"""三组预先固定随机样本的真实权重单帧候选诊断；零LLM请求、零实际执行。"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
from experiments.comparison.e1_protocol import group_overrides
from policies.diffusion_model_wrapper import DiffusionModelWrapper
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from simulator import Simulator
from single_actor_anchor_selection import select_single_actor

OUT=ROOT/'experiments/skills_single_actor_candidate_probe_scene182_20261020_r1'
SEEDS=[182000588,182000589,182000590]


def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise RuntimeError('probe_requires_reserved_idle_gpu3')
    OUT.mkdir(exist_ok=False)
    source=ROOT/'experiments/skills_stage_d_paired_scene182_bailian_20261020_r1/skills_expanded/movies/scenario_182/execution_trace.jsonl'
    rows=[json.loads(line) for line in source.open()]
    plan=next(r['validated_plan'] for r in rows if r['kind']=='llm_output' and r['step']==0)
    if not plan or not plan.get('attack'):raise RuntimeError('fixed_probe_plan_not_available')
    manifest=json.loads((source.parents[3]/'manifest.json').read_text())
    # source.parents[3]为根冒烟目录；固定输入和计划，不搜索更有利的场景。
    dump('launch.json',dict(seeds=SEEDS,scene=182,plan=plan,source_trace=str(source),
        source_trace_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),LLM_calls=0,actual_executed_frames=0,
        scope='one native forward batch per seed; counterfactual selection only; not closed-loop or exact replay of full-run RNG',started_at=time.time()))
    def forbidden(*args,**kwargs):raise AssertionError('LLM_API_forbidden_in_native_candidate_probe')
    LLMAdversarialPlanner._request_attack_plan=forbidden
    overrides=group_overrides('riskweaver_profile_off','idm',42,400,'qwen3.8-flash',['qwen3.8-flash'])
    overrides=['sim.visualize=false' if x=='sim.visualize=true' else x for x in overrides]
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'cfgs')):
        cfg=compose(config_name='config',overrides=overrides)
    (OUT/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True),encoding='utf8')
    env=Simulator(cfg);env.test_files=manifest['scenario_files']
    original=DiffusionModelWrapper._select_joint_guided_action
    records=[];current_seed=None
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
        record=dict(seed=current_seed,actors=int(positions.shape[0]),samples=int(positions.shape[1]),
            native_index=native_index,proposed_index=index,native=native_metrics,proposed=proposed_metrics,
            native_anchor_audit=native_audit,proposed_anchor_audit=audit,
            anchor_score_weight=wrapper.anchor_candidate_score_weight,
            stage='before downstream dynamics projection and execution gates; no action executed')
        records.append(record)
        data=policy_net._prepare_guidance_data(model_batch)
        transform=data['world_from_agent'][0].detach().cpu().numpy()
        np.savez_compressed(OUT/f'candidates_seed_{current_seed}.npz',
            positions_local=positions.detach().cpu().numpy(),
            targets_local=targets.detach().cpu().numpy(),mask=mask.detach().cpu().numpy(),
            native_local=native.positions.detach().cpu().numpy(),proposed_local=proposed.positions.detach().cpu().numpy(),
            world_from_agent=transform,anchors_global=np.asarray(plan['anchors']),dt=.1)
        wrapper._last_anchor_audit=native_audit
        # 本诊断仍返回原动作；修正动作只作为反事实数据，不进入仿真执行。
        return native,native_index
    DiffusionModelWrapper._select_joint_guided_action=compare
    try:
        for seed in SEEDS:
            current_seed=seed;np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            env.reset(0)
            env.set_attack_intent(target_id=plan['attack_target_id'],anchors=plan['anchors'],strategy=plan['strategy'])
            env.attack_intent.update(duration=plan.get('duration',10),request_id=plan.get('request_id'))
            env.prepare_background_traffic()
            dump('probe_result.partial.json',dict(records=records,LLM_calls=0,actual_executed_frames=0))
    finally:DiffusionModelWrapper._select_joint_guided_action=original
    assert len(records)==3
    dump('probe_result.json',dict(records=records,LLM_calls=0,actual_executed_frames=0,finished_at=time.time()))
    dump('exit_status.json',dict(exit_code=0,finished_at=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():dump('exit_status.json',dict(exit_code=1,error_type=type(error).__name__,finished_at=time.time()))
        raise
