import csv
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def main(base_text):
    base=Path(base_text)
    manifest=json.loads((base/'scenario_manifest.json').read_text(encoding='utf-8'))
    rows=[]
    for index,source in enumerate(manifest['scenario_files']):
        trace=base/'screen'/'movies'/f'scenario_{index:03d}'/'execution_trace.jsonl'
        records=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
        for record in records:
            if record.get('kind')!='profile_ranking_input':
                continue
            context=record['context']
            candidates=context.get('candidates',[])
            evidence=[int(c.get('profile_evidence_frames',0)) for c in candidates]
            rows.append({
                'scenario_index':index,
                'scene':Path(source).name,
                'step':int(record['step']),
                'candidate_count':len(candidates),
                'target_ids':sorted({int(c['target_id']) for c in candidates}),
                'strategies':[c['strategy'] for c in candidates],
                'max_profile_evidence_frames':max(evidence,default=0),
                'candidate_rejections':context.get('candidate_rejections',{}),
            })
    (base/'candidate_screen_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    with (base/'candidate_screen_summary.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['scenario_index','scene','step','candidate_count','target_ids','strategies','max_profile_evidence_frames','candidate_rejections'])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row,'target_ids':json.dumps(row['target_ids']),'strategies':json.dumps(row['strategies'],ensure_ascii=False),'candidate_rejections':json.dumps(row['candidate_rejections'],ensure_ascii=False,sort_keys=True)})
    fonts=list(font_manager.findSystemFonts(fontpaths=['/usr/share/fonts/opentype/noto']))
    chosen=next((f for f in fonts if 'CJK' in f and ('Regular' in f or 'DemiLight' in f)),None)
    if chosen:
        font_manager.fontManager.addfont(chosen)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=chosen).get_name()
    scenes=[]
    for row in rows:
        if row['scene'] not in scenes: scenes.append(row['scene'])
    steps=sorted({row['step'] for row in rows})
    matrix=np.full((len(scenes),len(steps)),np.nan)
    for row in rows: matrix[scenes.index(row['scene']),steps.index(row['step'])]=row['candidate_count']
    fig,ax=plt.subplots(figsize=(max(7,len(steps)*1.4),max(4,len(scenes)*.65+1.8)),constrained_layout=True)
    im=ax.imshow(matrix,aspect='auto',cmap='YlOrRd',vmin=0,vmax=max(1,np.nanmax(matrix)))
    for i in range(len(scenes)):
        for j in range(len(steps)):
            if np.isfinite(matrix[i,j]): ax.text(j,i,str(int(matrix[i,j])),ha='center',va='center',color='black',fontsize=10)
    ax.set_xticks(range(len(steps)),[f'第 {s} 帧' for s in steps])
    ax.set_yticks(range(len(scenes)),scenes)
    ax.set_xlabel('无攻击闭环中的画像规划点')
    ax.set_ylabel('冻结场景')
    ax.set_title('画像候选持续性预筛（不调用 LLM、不注入攻击）')
    cbar=fig.colorbar(im,ax=ax,pad=.02)
    cbar.set_label('通过规则初筛的候选数')
    fig.savefig(base/'画像候选持续性预筛.png',dpi=180)
    print(json.dumps({'rows':len(rows),'scenes':len(scenes),'positive_scenes':sum(any(r['scene']==s and r['candidate_count']>0 for r in rows) for s in scenes)},ensure_ascii=False))

if __name__=='__main__':
    main(sys.argv[1])
