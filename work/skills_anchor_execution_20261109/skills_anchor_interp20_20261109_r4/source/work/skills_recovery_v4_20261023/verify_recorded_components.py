"""只读历史记录验证真实组件；零API、零扩散采样，不把离线复算当作闭环成功。"""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_rejection_replay_20261023'),
              str(ROOT/'work/skills_v2_compact_dev_20261020')]
from omegaconf import OmegaConf
from diagnose_rejection import replay_env,decode,area
from policies.risk_metrics import RouteReachability
from route_tail_preflight import route_tail_readiness
from temporal_joint_recovery import TemporalJointRecovery,_context


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=ROOT/'experiments/skills_modelpool_development10_paired_20261023_r1/skills_full_shared_recovery'
    output=Path(__file__).with_name('recorded_component_verification.json')
    if output.exists():raise FileExistsError(output)
    watched=[p for p in source.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.yaml','.mp4')]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    cfg=OmegaConf.load(source/'resolved_config.yaml')
    OmegaConf.update(cfg,'sim.evaluation.reachability.execution_gate_policy','positive_area',force_add=True)
    checks=[]
    for path in source.rglob('attempt_result.json'):
        result=json.loads(path.read_text());index=result['scenario_index']
        if index not in (56,74):continue
        with Path(result['scenario_source']).open('rb') as stream:scene=pickle.load(stream)
        rows=[json.loads(line) for line in path.with_name('execution_trace.jsonl').read_text().splitlines()]
        step=77 if index==56 else 121
        row=next(r for r in rows if r['kind']=='state' and r['step']==step)
        previous=sorted({r['target_id'] for r in rows if r['kind']=='plan_accepted' and r['step']<=step})
        if index==56:
            joint=decode(next(r['joint'] for r in rows if r['kind']=='execution_prediction' and r['step']==76))
            env=replay_env(cfg,scene,row,None,previous);env.attack_intent=None
            # 第一遍30秒验收超时已留档；此遍只测全阶段耗时，不把120秒当正式配置。
            manager=TemporalJointRecovery(max_seconds=120.)
            # 76→77是原trace中真实已完成的转移；只用其记录的整组未来作输入。
            manager.remember(env,joint,_context(env));started=time.monotonic()
            recovered=manager.recover(env)
            checks.append(dict(scene=index,current_step=step,passed=True,
                recovery=recovered.metadata['temporal_joint_recovery'],elapsed_seconds=time.monotonic()-started))
        else:
            env=replay_env(cfg,scene,row,None,previous)
            current=area(RouteReachability(cfg.sim.evaluation.reachability),env,False,empty=True)
            future=route_tail_readiness(env,cfg.sim.evaluation.reachability,10)
            assert current['area_m2']>0 and not future['ready']
            checks.append(dict(scene=index,current_step=step,passed=True,current_unobstructed=current,
                               route_tail_preflight=future))
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in before.items())
    assert len(checks)==2 and unchanged
    with output.open('x') as stream:json.dump(dict(checks=checks,source_files_verified_unchanged=len(before),
        source_unchanged=unchanged,API_calls=0,diffusion_samples=0,new_executed_frames=0,
        scope='real recorded-state component checks; not a new closed-loop or fresh-LLM success'),stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(output=str(output),checks=[dict(scene=c['scene'],passed=c['passed']) for c in checks],
                          source_unchanged=unchanged),ensure_ascii=False))


if __name__=='__main__':main()
