"""零网络的固定意图生命周期消融；不是新的LLM成功调用或生成覆盖结果。"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'experiments/skills_stage_d_paired_scene182_bailian_20261020_r1/skills_expanded/movies/scenario_182/execution_trace.jsonl'


def physics_hash(state):
    """仅绑定物理可见状态；新实验目录名称不应改变同一意图的几何验证。"""
    value={k:state[k] for k in ('current_step','ego_state','agents','route','static_obstacles') if k in state}
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class RecordedIntentPlanner:
    def __init__(self,expected_state,plan,duration):
        self.expected=physics_hash(expected_state);self.plan=copy.deepcopy(plan);self.duration=duration
        self.available=True;self.provider='recorded_intent_no_network';self.use_multimodal=False
        self.use_obstacles=False;self.allow_trajectory=True;self.attack_mode='trajectory_only'
        self.last_request_failed=False;self.last_error=None;self.last_trace={};self.model_name='fixed_logged_intent'
        self.model_names=('fixed_logged_intent',);self.client=None;self.calls=0

    def generate_attack_plan(self,state,*args,**kwargs):
        self.last_request_failed=False
        self.last_trace=dict(mode='fixed_logged_intent_replay',inference_calls=0,
                             source_model='qwen3.8-flash',source_trace=str(SOURCE),
                             declared_duration_frames=self.duration)
        if state['current_step']!=0 or self.calls:
            return dict(attack=False,attack_target_id=-1,strategy='none',anchors=[],reason='single_fixed_intent_no_additional_plan')
        if physics_hash(state)!=self.expected:
            raise RuntimeError('recorded_intent_physics_state_mismatch')
        self.calls+=1
        result=copy.deepcopy(self.plan);result['duration']=self.duration
        # 原调用身份作为来源引用保留；新执行ID明确标识重放，不能当作新API响应。
        result['request_id']=f'fixed-replay-duration-{self.duration}'
        return result


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--duration',type=int,choices=(10,30),required=True)
    args,remaining=parser.parse_known_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:raise ValueError('explicit device isolation required')
    rows=[json.loads(line) for line in SOURCE.open()]
    state=next(r['state'] for r in rows if r['kind']=='llm_input' and r['step']==0)
    plan=next(r['validated_plan'] for r in rows if r['kind']=='llm_output' and r['step']==0)
    output=Path(remaining[remaining.index('--output-dir')+1])
    if output.exists():raise FileExistsError(output)
    import torch
    torch.set_num_threads(2)
    import simulator
    from experiments.comparison import run_e1_group as runner
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    factory=simulator.LLMAdversarialPlanner;request=LLMAdversarialPlanner._request_attack_plan;overrides=runner.group_overrides
    def forbidden(*a,**k):raise AssertionError('all_LLM_network_requests_forbidden_in_fixed_intent_ablation')
    simulator.LLMAdversarialPlanner=lambda *a,**k:RecordedIntentPlanner(state,plan,args.duration)
    LLMAdversarialPlanner._request_attack_plan=forbidden
    def configured(*a,**k):
        values=overrides(*a,**k)
        values+=['sim.llm.skills.enabled=false',f'sim.scenario_data_output_path={output}/scenario_debug']
        if os.environ['CUDA_VISIBLE_DEVICES']=='':values+=['sim.traffic_model.device=cpu']
        return values
    runner.group_overrides=configured
    sys.argv=[sys.argv[0]]+remaining
    try:runner.main()
    finally:
        simulator.LLMAdversarialPlanner=factory;LLMAdversarialPlanner._request_attack_plan=request;runner.group_overrides=overrides
        if output.exists():
            (output/'fixed_intent_protocol.json').write_text(json.dumps(dict(
                scope='one previously accepted recorded intent; fixed plan duration ablation, not fresh LLM generation',
                source_trace=str(SOURCE),source_trace_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                initial_physics_hash=physics_hash(state),duration_frames=args.duration,inference_calls=0,
                weights=[1.5,1,2,4],solver=dict(inner_lr=.2,inner_beta=.5,n_guide_steps=2)),indent=2),encoding='utf-8')


if __name__=='__main__':main()
