"""进一步区分路线截断与CV预测阻塞；反事实只用于诊断，不能用于执行或改D。"""
from pathlib import Path
import json,pickle,sys,copy,os
import numpy as np
from omegaconf import OmegaConf
from diagnose_rejection import ROOT,replay_env,decode,area
from policies.risk_metrics import RouteReachability


def main():
    assert os.getenv('CUDA_VISIBLE_DEVICES')==''
    root=ROOT/'experiments/skills_modelpool_development10_paired_20261023_r1/skills_full_shared_recovery'
    cfg=OmegaConf.load(root/'resolved_config.yaml');out=[]
    for p in root.rglob('attempt_result.json'):
        a=json.loads(p.read_text());index=a['scenario_index']
        if index not in [74,181]:continue
        with Path(a['scenario_source']).open('rb') as f:scene=pickle.load(f)
        rows=[json.loads(x) for x in p.with_name('execution_trace.jsonl').read_text().splitlines()];step=a['executed_steps']
        row=next(x for x in rows if x['kind']=='state' and x['step']==step)
        j=decode(next(x['joint'] for x in rows if x['kind']=='execution_prediction' and x['step']==step))
        env=replay_env(cfg,scene,row,j,());reach=RouteReachability(cfg.sim.evaluation.reachability)
        if index==74:
            route,arc,tangent=reach._route_geometry(scene['route']);extended=np.vstack([route,route[-1]+np.arange(1,21)[:,None]*tangent[-1]])
            original=area(reach,env,False,empty=True)
            env.scenario_dict=dict(scene,route=extended)
            extended_area=area(reach,env,False,empty=True)
            item=dict(scene=index,step=step,original_no_traffic=original,diagnostic_20m_route_extension=extended_area,
                      warning='synthetic geometry diagnosis only; not evidence of actual road, not used for execution, D or success')
        else:
            original_active=env.agent_active.copy();checks=[]
            for actor in np.flatnonzero(original_active):
                env.agent_active=original_active.copy();env.agent_active[actor]=False
                checks.append(dict(removed_actor=int(actor),CV_area=area(reach,env,False)))
            env.agent_active=original_active
            item=dict(scene=index,step=step,CV_area=area(reach,env,False),recorded_joint_area=area(reach,env,True),
                      single_actor_removal=checks,current_ego=row['ego'],current_active_actor_states=[dict(id=int(i),state=row['agents'][i]) for i in np.flatnonzero(original_active)],
                      warning='actor removal is diagnosis only; no actor omitted in evaluation or execution')
        out.append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
    path=ROOT/'work/skills_rejection_replay_20261023/causality_checks.json'
    with path.open('x') as f:json.dump(dict(records=out,LLM_calls=0,diffusion_samples=0,executed_frames=0,original_results_unchanged=True),f,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
