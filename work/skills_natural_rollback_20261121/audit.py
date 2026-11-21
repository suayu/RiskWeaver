# 只读验收：实际攻击归属和危险证据；背景碰撞不否决，不当作自车碰撞。
from pathlib import Path
import json,subprocess
from policies.collision_ttc import minimum_obb_ttc

def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def ttc(ego,other):
    import math
    value=minimum_obb_ttc(ego,[other],[True])
    return value if math.isfinite(value) else None

def audit(case):
    out=case/'skills';p=out/'results.json'
    if not p.exists():return dict(danger=False,dynamic_danger=False,reason='missing_results')
    episodes=read(p)['episodes']
    if len(episodes)!=1:return dict(danger=False,dynamic_danger=False,reason='unexpected_episode_count')
    a=episodes[0];tr=lines(out/'movies'/f"scenario_{a['scenario_index']:03d}"/'execution_trace.jsonl')
    states={x['step']:x for x in tr if x.get('kind')=='state'}
    plans={x['validated_plan'].get('request_id'):x['validated_plan'] for x in tr if x.get('kind')=='llm_output' and isinstance(x.get('validated_plan'),dict)}
    created={};dynamic=[];static=[];invalid=[]
    for x in tr:
        if x.get('kind')=='static_attack_execution' and (plans.get(x.get('request_id'))or{}).get('obstacle_plan'):
            for ident in x.get('obstacle_ids',[]):created[ident]=x['step']
        if x.get('kind')=='attack_execution' and x.get('executed'):
            plan=plans.get(x.get('request_id'))or{};s=states.get(x['step']);target=x.get('target_id')
            if not s or not plan.get('attack') or plan.get('attack_target_id')!=target:invalid.append('dynamic_identity');continue
            value=ttc(s['ego'],s['agents'][target]);dynamic.append(dict(step=x['step'],target=target,TTC=value))
    for step,s in states.items():
        if step==0:continue
        for o in s.get('static_obstacles',[]):
            start=created.get(o['id'])
            if start is None or step<start:continue
            box=[*o['center'][:2],0.,0.,o['yaw'],o['length'],o['width']]
            if start not in states or ttc(states[start]['ego'],box)==0:invalid.append('static_initial_overlap');continue
            static.append(dict(step=step,target=o['id'],TTC=ttc(s['ego'],box)))
    def danger(rows):
        evidence=[x for x in rows if x['TTC'] is not None and x['TTC']<=3.]
        contact=any(x['TTC']==0 for x in evidence)
        return bool(evidence and (len({x['step']for x in rows})>=10 or contact))
    clean=a.get('status')=='completed' and not any(a.get(k) for k in ['runtime_failure','generation_failure','planner_service_failure_count','prediction_service_failure_count'])
    last=states.get(a.get('executed_steps'),{});info=last.get('info')or{}
    natural=bool(info.get('collision') or info.get('completed') or info.get('off_route'))
    valid_states=sorted(states)==list(range(a.get('executed_steps',0)+1))
    videos=list(out.rglob('*.mp4'))+list(case.glob('skills_integer_video_exports/*.mp4'));media=[]
    for v in videos:
        p=subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(v),'-f','null','-'],capture_output=True,timeout=120)
        probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','json',str(v)],capture_output=True,timeout=120)
        n=int(json.loads(probe.stdout)['streams'][0]['nb_read_frames']) if probe.returncode==0 else None
        media.append(dict(path=str(v),decoded=p.returncode==0,frames=n,matches_trace=n==len(states)))
    full=clean and natural and valid_states and not invalid and len(media)>=2 and all(x['decoded'] and x['matches_trace']for x in media)
    return dict(scenario_index=a['scenario_index'],danger=bool(full and (danger(dynamic)or danger(static))),
        dynamic_danger=bool(full and danger(dynamic)),clean=clean,natural_end=natural,executed_steps=a['executed_steps'],
        dynamic_frames=len(dynamic),static_exposure_frames=len({x['step']for x in static}),
        dynamic_evidence=dynamic,static_evidence=static,issues=sorted(set(invalid)),videos=media,
        ego_collision=a.get('collision'),background=a.get('feasibility'),
        I=a.get('scene_challenge_score'),R=a.get('scenario_danger_score'),danger_coverage=a.get('danger_coverage'),
        effect_statistics_eligible=clean,reason=None if full else 'incomplete_or_fault_or_non_natural')
