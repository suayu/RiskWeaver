import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def finite(value):
    try:
        value=float(value)
        return value if np.isfinite(value) else None
    except (TypeError,ValueError):
        return None


def read_group(base,name,manifest):
    rows=[]
    for index,source in enumerate(manifest['scenario_files']):
        movie=base/name/'movies'/f'scenario_{index:03d}'
        result=json.loads((movie/'attempt_result.json').read_text(encoding='utf-8'))
        trace=[json.loads(line) for line in (movie/'execution_trace.jsonl').read_text(encoding='utf-8').splitlines()]
        llm=[finite(r.get('planner_wall_seconds')) for r in trace if r.get('kind')=='llm_output']
        llm=[x for x in llm if x is not None]
        windows=[w for w in result.get('attack_window_difficulties',[]) if w.get('valid')]
        ttcs=[finite(w.get('metrics',{}).get('avg_min_ttc')) for w in windows]
        ttcs=[x for x in ttcs if x is not None]
        ranking=[r for r in trace if r.get('kind')=='profile_ranking_input']
        evidence=[int(c.get('profile_evidence_frames',0)) for r in ranking for c in r.get('context',{}).get('candidates',[])]
        feasibility=result.get('feasibility',{})
        rows.append({
            'group':name,'scenario_index':index,'scene':Path(source).name,
            'D':finite(result.get('scenario_danger_score')),'danger_valid':bool(result.get('danger_valid')),
            'attack_plan_count':int(result.get('attack_plan_count',0)),
            'attack_frames':int(result.get('attack_executed_frames',result.get('attack_active_frames',0))),
            'danger_per_attack_frame':(finite(result.get('scenario_danger_score'))/int(result.get('attack_executed_frames',result.get('attack_active_frames',0))) if finite(result.get('scenario_danger_score')) is not None and int(result.get('attack_executed_frames',result.get('attack_active_frames',0)))>0 else None),
            'min_valid_window_ttc_s':min(ttcs,default=None),
            'planner_service_failures':int(result.get('planner_service_failure_count',0)),
            'generation_failure':result.get('generation_failure'),
            'background_collision_frames':int(feasibility.get('background_collision_frames',0)),
            'background_static_collision_frames':int(feasibility.get('background_static_collision_frames',0)),
            'mean_llm_wall_seconds':float(np.mean(llm)) if llm else None,
            'max_profile_evidence_frames':max(evidence,default=0),
            'history_informed_candidate_count':sum(v>=20 for v in evidence),
            'collision':bool(result.get('collision')),
        })
    return rows


def ci(values,rng,draws=10000):
    values=np.asarray(values,float)
    if not len(values): return [None,None]
    boot=np.mean(rng.choice(values,(draws,len(values))),axis=1)
    return np.quantile(boot,[.025,.975]).tolist()


def main(base_text):
    base=Path(base_text)
    manifest=json.loads((base/'scenario_manifest.json').read_text(encoding='utf-8'))
    off=read_group(base,'profile_off',manifest)
    on=read_group(base,'profile_on',manifest)
    pairs=[]
    for a,b in zip(off,on):
        assert a['scene']==b['scene']
        pairs.append({'scene':a['scene'],'off_D':a['D'],'on_D':b['D'],
                      'D_difference':b['D']-a['D'] if a['D'] is not None and b['D'] is not None else None,
                      'off_attack_frames':a['attack_frames'],'on_attack_frames':b['attack_frames'],
                      'off_min_ttc':a['min_valid_window_ttc_s'],'on_min_ttc':b['min_valid_window_ttc_s'],
                      'max_profile_evidence_frames':b['max_profile_evidence_frames']})
    rng=np.random.default_rng(42)
    valid_diff=[p['D_difference'] for p in pairs if p['D_difference'] is not None]
    summary={
        'groups':{},'paired':{
            'valid_pairs':len(valid_diff),'mean_D_difference':float(np.mean(valid_diff)) if valid_diff else None,
            'mean_D_difference_bootstrap95':ci(valid_diff,rng),
            'profile_on_higher_D_pairs':sum(x>0 for x in valid_diff),
            'pairs':pairs,
        },
        'scope_warning':'six-scene paired pilot; directional engineering evidence, not final statistical claim',
    }
    for name,rows in [('profile_off',off),('profile_on',on)]:
        d=[r['D'] for r in rows if r['D'] is not None]
        summary['groups'][name]={
            'scenario_count':len(rows),'valid_D_count':len(d),'mean_D':float(np.mean(d)) if d else None,
            'attack_scene_count':sum(r['attack_frames']>0 for r in rows),'attack_frames':sum(r['attack_frames'] for r in rows),
            'plan_count':sum(r['attack_plan_count'] for r in rows),
            'service_failures':sum(r['planner_service_failures'] for r in rows),
            'generation_failures':sum(bool(r['generation_failure']) for r in rows),
            'background_collision_frames':sum(r['background_collision_frames'] for r in rows),
            'background_static_collision_frames':sum(r['background_static_collision_frames'] for r in rows),
            'mean_llm_wall_seconds':float(np.mean([r['mean_llm_wall_seconds'] for r in rows if r['mean_llm_wall_seconds'] is not None])),
            'history_informed_candidate_count':sum(r['history_informed_candidate_count'] for r in rows),
        }
    (base/'paired_summary.json').write_text(json.dumps({'summary':summary,'rows':off+on},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with (base/'paired_scene_metrics.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list((off+on)[0]))
        writer.writeheader(); writer.writerows(off+on)
    fonts=list(font_manager.findSystemFonts(fontpaths=['/usr/share/fonts/opentype/noto']))
    chosen=next((f for f in fonts if 'CJK' in f and ('Regular' in f or 'DemiLight' in f)),None)
    if chosen:
        font_manager.fontManager.addfont(chosen); plt.rcParams['font.family']=font_manager.FontProperties(fname=chosen).get_name()
    labels=[p['scene'].replace('.pkl','') for p in pairs]
    x=np.arange(len(labels)); width=.36
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    axes[0].bar(x-width/2,[p['off_D'] or 0 for p in pairs],width,label='画像关闭',color='#8faadc')
    axes[0].bar(x+width/2,[p['on_D'] or 0 for p in pairs],width,label='画像开启',color='#ed7d31')
    axes[0].set_xticks(x,labels,rotation=30); axes[0].set_ylabel('场景危险度 D'); axes[0].set_title('逐场景危险度'); axes[0].legend()
    axes[1].bar(x-width/2,[p['off_attack_frames'] for p in pairs],width,label='画像关闭',color='#8faadc')
    axes[1].bar(x+width/2,[p['on_attack_frames'] for p in pairs],width,label='画像开启',color='#ed7d31')
    axes[1].set_xticks(x,labels,rotation=30); axes[1].set_ylabel('真实攻击执行帧'); axes[1].set_title('攻击执行覆盖')
    diffs=[p['D_difference'] if p['D_difference'] is not None else 0 for p in pairs]
    axes[2].bar(x,diffs,color=['#c00000' if d>0 else '#70ad47' for d in diffs]); axes[2].axhline(0,color='black',linewidth=.8)
    axes[2].set_xticks(x,labels,rotation=30); axes[2].set_ylabel('D(画像开启) − D(画像关闭)'); axes[2].set_title('配对效应方向')
    fig.suptitle('自车画像与针对性攻击：六场景配对先导实验',fontsize=15)
    fig.savefig(base/'自车画像六场景配对先导.png',dpi=180)
    print(json.dumps(summary,ensure_ascii=False,allow_nan=False))

if __name__=='__main__': main(sys.argv[1])
