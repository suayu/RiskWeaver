"""无LLM、无GPU地重放既有40次画像规划输入；保留原始空目录。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_ego_profile import planner as fixture_planner
from policies.skills_planner import SkillsAttackPlanner
from policies.intent_validation import validate_world_intent

SOURCE=ROOT/'experiments/e1_five_arm_idm_10_modelpool_20261014_r1/formal/riskweaver_profile_on'
OUT=ROOT/'experiments/skills_stage_a_replay_20261020_r1'
OUT.mkdir(exist_ok=False)
records=[]
for path in sorted(SOURCE.rglob('execution_trace.jsonl')):
    rows=[json.loads(line) for line in path.open()]
    states={r['step']:r for r in rows if r['kind']=='state'}
    contexts={r['step']:r['context'] for r in rows if r['kind']=='profile_ranking_input'}
    for request in [r for r in rows if r['kind']=='llm_input']:
        step=request['step'];state=request['state'];frame=states[step]
        route=np.asarray(state['route'],float)
        env=NS(ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
               agent_active=np.asarray(frame['active']),current_step=step,current_scene_id=path.parent.name,
               scenario_dict={'route':route,'lanes':[route]},get_static_obstacles=lambda:[])
        env.get_state_for_planning=lambda state=state:state
        p=fixture_planner();p.use_obstacles=False;p.provider='dashscope'
        original=contexts[step]
        record=dict(scene=path.parent.name,step=step,legacy_candidates=len(original['candidates']),
                    legacy_rejections=original.get('candidate_rejections'),modes={})
        for expansion in (False,True):
            engine=SkillsAttackPlanner(NS(llm=NS(skills=NS(candidate_expansion_enabled=expansion)),ego_profile=NS()))
            context=engine.build_context(env,state,p,profile_context=original)
            checks=[validate_world_intent(p,state,c) for c in context['candidates']]
            assert all(c['accepted'] for c in checks)
            record['modes'][str(expansion)]=dict(candidate_count=len(checks),precheck=context['semantic_precheck'],
                                                  candidates=context['candidates'])
        records.append(record)
summary=dict(planning_opportunities=len(records),legacy_nonempty=sum(r['legacy_candidates']>0 for r in records),
    unified_nonempty=sum(r['modes']['False']['candidate_count']>0 for r in records),
    expanded_nonempty=sum(r['modes']['True']['candidate_count']>0 for r in records),
    semantic_publish_final_consistency=True,LLM_calls=0,
    limitations=['Observed states and original catalogs are exact; road polyline is the request route. Road topology is not an attacker screen here.','CV background screening is a proxy, not diffusion or executed safety.','Development replay; no claim of closed-loop attack coverage.'])
(OUT/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False))
