"""固定十个初始状态的百炼技能消融；零GPU，不把计划通过当作实际攻击。"""
import copy
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path('/home2/zhaoyx/scenario-dreamer');sys.path.insert(0,str(ROOT))
from policies.skills_planner import SkillsAttackPlanner
from policies.llm_adversarial_planner import LLMAdversarialPlanner

OUT=ROOT/'experiments/skills_stage_b_frozen10_bailian_20261020_r1'
MODEL='qwen3.8-flash'
ARMS=[('legacy_free',None,False),('catalog_no_knowledge','none',True),
      ('protocol','protocol',True),('analysis','analysis',True),('full','full',True),('full_no_expansion','full',False)]
SOURCE=ROOT/'experiments/e1_five_arm_idm_10_modelpool_20261014_r1/formal/riskweaver_profile_off'
INSTRUCTION=['通过背景交通参与者的切入、前向压迫、合流冲突等攻击性行为，挑战自车驾驶策略。仅选择离散攻击策略、目标及稀疏锚点，禁止放置静态障碍物。']


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf8')


def main():
    OUT.mkdir(exist_ok=False)
    dump(OUT/'launch.json',dict(provider='dashscope',model=MODEL,arms=ARMS,planned=60,
        scene_selection='same frozen development10 first pre-attack state',max_calls_per_arm_state=1,
        expansion_confound_separate=True,skills_not_closed_loop=True,started_at=time.time()))
    records=[]
    for scene_no,path in enumerate(sorted(SOURCE.rglob('execution_trace.jsonl'))):
        rows=[json.loads(line) for line in path.open()]
        request=next(r for r in rows if r['kind']=='llm_input');frame=next(r for r in rows if r['kind']=='state' and r['step']==request['step'])
        state=request['state'];route=np.asarray(state['route'],float)
        env=NS(ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
            agent_active=np.asarray(frame['active']),current_step=request['step'],current_scene_id=path.parent.name,
            scenario_dict={'route':route,'lanes':[route]},get_static_obstacles=lambda:[])
        env.get_state_for_planning=lambda state=state:state
        # 平衡不同方法在服务时间上的顺序；状态、模型与总调用预算保持一致。
        order=ARMS[scene_no%len(ARMS):]+ARMS[:scene_no%len(ARMS)]
        for arm,level,expansion in order:
            record=dict(scene=path.parent.name,step=request['step'],arm=arm,model=MODEL,started_at=time.time())
            planner=LLMAdversarialPlanner(model_name=MODEL,model_names=[MODEL],provider='dashscope',attack_mode='trajectory_only')
            if not planner.available:raise RuntimeError('bailian_not_configured')
            transport=[];create=planner.client.chat.completions.create
            def traced(*a,**kw):
                event=dict(messages=kw.get('messages'),model=kw.get('model'));transport.append(event);start=time.monotonic()
                try:
                    response=create(*a,**kw)
                    event.update(content=response.choices[0].message.content,usage=response.usage.model_dump() if response.usage else None,model_returned=response.model)
                    return response
                except Exception as error:
                    event.update(error_type=type(error).__name__,status_code=getattr(error,'status_code',None));raise
                finally:event['elapsed_seconds']=time.monotonic()-start
            planner.client.chat.completions.create=traced
            start=time.monotonic()
            if level is None:
                plan=planner.generate_attack_plan(copy.deepcopy(state),INSTRUCTION)
            else:
                engine=SkillsAttackPlanner(NS(llm=NS(skills=NS(level=level,candidate_expansion_enabled=expansion,max_repairs=0)),ego_profile=NS()),1)
                plan=engine.plan(planner,env,copy.deepcopy(state),INSTRUCTION)
            record.update(elapsed_seconds=time.monotonic()-start,plan=plan,trace=planner.last_trace,
                service_failure=planner.last_request_failed,semantic_attack_accepted=bool(plan and plan.get('attack')),transport=transport)
            records.append(record)
            dump(OUT/f'{path.parent.name}_{arm}.json',record)
            summary={name:dict(attempts=sum(r['arm']==name for r in records),
                        attack_plans=sum(r['arm']==name and r['semantic_attack_accepted'] for r in records),
                        api_requests=sum(len(r['transport']) for r in records if r['arm']==name),
                        service_failures=sum(r['arm']==name and r['service_failure'] for r in records)) for name,_,_ in ARMS}
            dump(OUT/'summary.partial.json',summary)
            if planner.last_request_failed:
                raise RuntimeError('final_planner_service_failure; frozen model not silently switched')
    dump(OUT/'summary.json',summary)
    dump(OUT/'exit_status.json',dict(exit_code=0,finished_at=time.time(),records=len(records)))


if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():dump(OUT/'exit_status.json',dict(exit_code=1,error_type=type(error).__name__,finished_at=time.time()))
        raise
