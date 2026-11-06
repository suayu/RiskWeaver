"""回放开发轨迹的后期候选机会；不调用模型，也不冒充闭环反事实结果。"""
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from policies.skills_planner import SkillsAttackPlanner
from test_ego_profile import planner as fixture_planner

OUT=ROOT/'experiments/skills_late_opportunity_replay_dev10_20261020_r1'
SOURCE=ROOT/'experiments/e1_five_arm_idm_10_modelpool_20261014_r1/formal/riskweaver_profile_off'


def main():
    OUT.mkdir(exist_ok=False)
    records=[]
    for path in sorted(SOURCE.rglob('execution_trace.jsonl')):
        rows=[json.loads(line) for line in path.open()]
        frames={r['step']:r for r in rows if r['kind']=='state'}
        attempt=json.loads((path.parent/'attempt_result.json').read_text())
        with open(attempt['scenario_source'],'rb') as stream:scene=pickle.load(stream)
        route=np.asarray(scene['route'],float)
        observations=[]
        engine=SkillsAttackPlanner(NS(llm=NS(skills=NS(candidate_expansion_enabled=True,max_repairs=0)),ego_profile=NS()),4)
        planner=fixture_planner();planner.use_obstacles=False
        for step in range(0,max(frames)+1,11):
            frame=frames[step];ego=np.asarray(frame['ego']);states=np.asarray(frame['agents']);active=np.asarray(frame['active'])
            assert not frame['static_obstacles']
            visible=[int(i) for i in np.flatnonzero(active) if np.linalg.norm(states[i,:2]-ego[:2])<=35.]
            state=dict(ego_state=ego[:5].tolist(),route=route[np.linalg.norm(route[:,:2]-ego[:2],axis=1)<=35.,:2].tolist(),
                agents=[dict(id=i,state=states[i,:5].tolist(),type='vehicle',history=[]) for i in visible],
                current_step=step,history_order='oldest_to_newest',static_obstacles=[])
            env=NS(ego_state=ego,data_dict={'agent':[states]},agent_active=active,current_step=step,
                current_scene_id=path.parent.name,scenario_dict=dict(scene,route=route),get_static_obstacles=lambda:[])
            env.get_state_for_planning=lambda state=state:state
            context=engine.build_context(env,state,planner)
            observations.append(dict(step=step,visible_ids=visible,candidate_count=len(context['candidates']),
                                     targets=sorted({c['target_id'] for c in context['candidates']})))
        early=[x for x in observations if x['step']<=33]
        late=[x for x in observations if x['step']>33]
        records.append(dict(scene=path.parent.name,observed_execution_frames=attempt['executed_steps'],
            early_nonempty=sum(x['candidate_count']>0 for x in early),late_nonempty=sum(x['candidate_count']>0 for x in late),
            observations=observations))
    summary=dict(scenes=len(records),early_candidate_scene_coverage=sum(r['early_nonempty']>0 for r in records)/len(records),
        anywhere_observed_candidate_scene_coverage=sum(r['early_nonempty']+r['late_nonempty']>0 for r in records)/len(records),
        extra_late_only_scenes=[r['scene'] for r in records if r['early_nonempty']==0 and r['late_nonempty']>0],
        late_opportunity_scenes=[r['scene'] for r in records if r['late_nonempty']>0],
        LLM_calls=0,GPU_calls=0,scope='observational development replay; histories omitted as candidate code does not use them; nearby radius35m matches simulator; no claims about actual model choices or counterfactual state evolution')
    (OUT/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
