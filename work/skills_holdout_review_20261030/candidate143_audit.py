"""只读143冻结状态与现有验证器；展示不同前置门槛，不改变目录或启动仿真/API。"""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,json,os,sys
import numpy as np
from omegaconf import OmegaConf
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.intent_validation import validate_world_intent
from policies.ego_profile import CandidateBuilder,JointTrajectorySafetyGate

ROOT=Path('/home2/zhaoyx/scenario-dreamer')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only_required')
    data=ROOT/'experiments/skills_candidate_local_highlevel_holdout10_20261030_r1'
    input_file=data/'inputs.json';samples=json.loads(input_file.read_text());sample=next(s for s in samples if s['scene']==143)
    cfg=OmegaConf.load(data/'resolved_config.yaml');state=sample['state'];frame=sample['frame']
    env=NS(ego_state=np.array(frame['ego']),data_dict={'agent':[np.array(frame['agents'])]},agent_active=np.array(frame['active']),get_static_obstacles=lambda:[])
    actor=state['agents'][0];ego=np.array(state['ego_state']);a=np.array(actor['state']);T=3.
    base=a[:2]+T*a[2:4];future=ego[:2]+T*ego[2:4];direction=(future-base)/np.linalg.norm(future-base)
    planner=object.__new__(LLMAdversarialPlanner)  # 仅使用无服务依赖的normalize/validate方法。
    builder=CandidateBuilder(cfg.sim.ego_profile);rows=[]
    for cap in (3.8,4.,5.):
        time=np.arange(4.)[:,None];anchors=a[:2]+time*a[2:4]+.5*time*time*cap*direction
        candidate=dict(candidate_id='diagnostic_cap_'+str(cap),target_id=actor['id'],strategy='lane_change',duration=10,anchors=anchors.tolist())
        semantic=validate_world_intent(planner,state,candidate)
        screening,reason=builder.screen(env,actor['id'],anchors,[])
        rows.append(dict(cap_mps2=cap,anchors=anchors.tolist(),semantic_validation=semantic,
            candidate_screen_passed=reason is None,candidate_screen_reason=reason,screening=screening,
            new_LLM_plan=False,execution_approved=False))
    result=dict(scene=143,source_input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
        published_candidate_count=len(sample['contexts']['temporal_original']['candidates']),
        original_candidate_audit=sample['contexts']['temporal_original']['candidate_audit'],
        current_execution_gate_limits=JointTrajectorySafetyGate(cfg.sim).dynamic_limits,
        current_screen_source_sha256=hashlib.sha256((ROOT/'policies/ego_profile.py').read_bytes()).hexdigest(),
        diagnostics=rows,actual_execution_frames=0,API_requests=0,D=None,
        scope='sparse_intent_prerequisite_diagnostic_not_continuous_feasibility_or_effectiveness_certificate',
        current_RAG_experiment_changed=False,original_holdout_results_unchanged=True)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
