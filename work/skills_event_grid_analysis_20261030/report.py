"""全部10初态与全部20规划尝试的透明报告；不认证自由文字或实际攻击收益。"""
from pathlib import Path
import csv,hashlib,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path('D:/RiskWeaver/outputs/skills_event_grid_20261030')
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    offline=OUT/'work/skills_event_grid_20261030/development10_initial_contexts_20261030_r1'
    rows=load(offline/'results.json')['records'];table=''
    counts=[]
    for scene in [53,56,74,80,103,140,181,182,207,220]:
        values={r['mode']:r for r in rows if r['scene']==scene};counts.append([values[m]['visible_candidate_count'] for m in ['temporal_original','full_grid']])
        table+='<tr>'+''.join(f'<td>{v}</td>' for v in [scene,*counts[-1],len(values['full_grid']['visible_event_ids'])])+'</tr>'
    grid=load(OUT/'work/skills_future_event_probe_20261029/scene074_all30_events_20261030_r1.json')['records']
    for lang in ['zh','en']:
        fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        times=[r['requested_event_seconds'] for r in grid];norms=[r['required_sparse_acceleration_norm_mps2'] for r in grid]
        axes[0].plot(times,norms,'o-',color='#426f9b',markersize=3)
        axes[0].axhline(3.8,color='#c54a3f',linestyle='--',label='原系数上限3.8' if lang=='zh' else 'Original coefficient cap 3.8')
        for r in grid:
            if r.get('all_original_prechecks_passed'):axes[0].scatter(r['requested_event_seconds'],r['required_sparse_acceleration_norm_mps2'],color='#24845d',s=65,zorder=4)
        axes[0].set_ylim(0,20);axes[0].set_xlim(.05,3.05)
        axes[0].set_title('74号初态：遗漏的未来交互时刻' if lang=='zh' else 'Scene 74: missed future event times',fontproperties=FONT if lang=='zh' else None)
        axes[0].set_xlabel('未来交互时刻（秒）' if lang=='zh' else 'Future event time (s)',fontproperties=FONT if lang=='zh' else None)
        axes[0].set_ylabel('所需稀疏加速度系数（米/秒²）' if lang=='zh' else 'Required sparse acceleration coefficient (m/s²)',fontproperties=FONT if lang=='zh' else None)
        axes[0].legend(prop=FONT if lang=='zh' else None);axes[0].grid(alpha=.2)
        x=np.arange(10);counts_array=np.asarray(counts)
        axes[1].bar(x-.18,counts_array[:,0],width=.36,label='原时序技能目录' if lang=='zh' else 'Temporal catalog')
        axes[1].bar(x+.18,counts_array[:,1],width=.36,label='完整事件网格' if lang=='zh' else 'Full event grid')
        axes[1].set_xticks(x,[53,56,74,80,103,140,181,182,207,220]);axes[1].set_ylim(0,27)
        axes[1].set_title('全部10初态：非空覆盖均7/10' if lang=='zh' else 'All 10 initial states: both nonempty in 7/10',fontproperties=FONT if lang=='zh' else None)
        axes[1].set_xlabel('冻结场景编号' if lang=='zh' else 'Frozen scene ID',fontproperties=FONT if lang=='zh' else None)
        axes[1].set_ylabel('可见候选数（上限24）' if lang=='zh' else 'Visible candidates (cap 24)',fontproperties=FONT if lang=='zh' else None)
        axes[1].legend(prop=FONT if lang=='zh' else None)
        fig.savefig(OUT/f'event_grid_diagnostic_{lang}.png',dpi=160);plt.close(fig)
    body=f'''<h1>未来交互时刻技能：开发诊断</h1><p>当前只使用原开发10个初态，不是独立泛化集。原时序技能与完整事件网格两臂都开启skills；该对照隔离新配方，不能称旧skills关闭/开启比较。</p><p>74初态在1.5/1.6/1.7秒通过原预检，旧0.5/1/3秒配方遗漏。网格配方在5场进入可见目录，但全部10初态的非空覆盖仍均为7/10，未证明候选覆盖提升。</p><img src="event_grid_diagnostic_zh.png"><table><tr><th>场景</th><th>原时序目录</th><th>完整网格目录</th><th>可见新配方数</th></tr>{table}</table>'''
    body+='<p>图左为局部低值放大，显示上限20；完整30个时刻的所有数值（含大于20和被拒绝者）保留在原JSON。稀疏系数不等于实测车辆加速度，CV相交不等于实际威胁或安全保证。</p>'
    run=OUT/'experiments/skills_event_grid_initial_planning10_paired_20261030_r1'
    if (run/'results.json').exists():
        result=load(run/'results.json');records=result['results'];summary={};trial_rows='';review=[]
        for mode in ['temporal_original','full_grid']:
            arm=[r for r in records if r['mode']==mode];physical=[]
            for r in arm:
                folder=run/f'scene{r["scene"]:03d}_{mode}';journal=folder/'api_calls.jsonl'
                if journal.exists():physical.extend([json.loads(x) for x in journal.read_text(encoding='utf-8').splitlines()])
            models={m:sum(p.get('model_returned')==m for p in physical) for m in sorted({p['model_returned'] for p in physical if p.get('model_returned')})}
            summary[mode]=dict(planned_states=10,attempted_states=len(arm),unattempted_states=10-len(arm),accepted_plan_states=sum(r['accepted_plan'] for r in arm),
                plan_coverage_per_planned_state=sum(r['accepted_plan'] for r in arm)/10,service_failure_count=sum(bool(r.get('planner_service_failure')) for r in arm),
                runtime_failure_count=sum(bool(r.get('runtime_failure')) for r in arm),physical_requests=len(physical),successful_physical_requests=sum('content' in p for p in physical),
                tokens=sum((p.get('usage') or {}).get('total_tokens',0) for p in physical),SDK_seconds=sum(p['elapsed_seconds'] for p in physical),models=models,
                actual_attack_frames=0,D=None,correct_plan_coverage=None,free_text_semantic_review='not_reviewed')
        for r in records:
            folder=run/f'scene{r["scene"]:03d}_{r["mode"]}';trace=r.get('planner_trace') or {};context=trace.get('context') or {}
            response=trace.get('response_summary') or {};selected=r.get('selected_candidate');catalog={c['candidate_id']:c for c in context.get('candidates',[])}
            status='接收' if r['accepted_plan'] else (r.get('validation_reason') or trace.get('plan_validation'))
            trial_rows+=f'<tr><td>{r["scene"]}</td><td>{r["mode"]}</td><td>{html.escape(str(status))}</td><td>{html.escape(str(selected))}</td><td>{r["budget"]["transport_calls"]}</td><td><a href="{(folder/"result.json").relative_to(OUT).as_posix()}">完整请求回复与编译记录</a></td></tr>'
            for request in trace.get('requests',[]):
                response_i=request.get('response') or {}
                review.append(dict(scene=r['scene'],mode=r['mode'],repair=request.get('repair'),selected_candidate=selected,
                    scene_analysis=response_i.get('scene_analysis'),candidate_reasons=response_i.get('candidate_reasons'),ranking=response_i.get('ranking'),
                    grounding=response_i.get('grounding'),candidate_facts={cid:dict(candidate=catalog.get(cid),
                        motion=context.get('temporal_candidate_evidence',{}).get('current_source_motion',{}),
                        priority=context.get('priority_evidence',{}).get(cid),contrast=context.get('incremental_cards',{}).get(cid),
                        future=context.get('rolling_intent',{}).get('candidate_facts',{}).get(cid)) for cid in response_i.get('ranking',[])},
                    review_status='not_reviewed',prose_or_tactics_correct=None))
        (OUT/'planning_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/'all_planning_claims_for_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
        body+='<h2>真实百炼高层规划结果</h2><p>每臂10个预定初态，包含无目录、失败和未尝试；每状态最多4次物理请求。无GPU、无新仿真、无实际攻击帧，D保持null。JSON/接收不等于全部文字或战术正确，正确规划覆盖尚未认证。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
        body+='<p><a href="all_planning_claims_for_review.json">全部场景分析、全部排名理由及核验事实</a></p><table><tr><th>场景</th><th>方法</th><th>状态</th><th>选择</th><th>物理请求</th><th>原始记录</th></tr>'+trial_rows+'</table>'
    else:body+='<h2>真实百炼高层规划</h2><p>已冻结321源码和全部10初态两臂，共20预定，正在独立运行。终态前不发布部分性能结论。该任务未修改正在运行的311源码闭环组。</p>'
    body+='<h2>交付边界</h2><p>旧技能/调用路径保留。新事件技能默认关闭，入口run_event_controlled.py --future-event-grid off|on。不修改扩散、背景安全或动力学阈值；新模块测试和真实prepare-only不等于多数正确攻击。当前目标未完成，原A–E、留出与多模式验证继续保留。本轮纯高层诊断没有新视频。</p><p><a href="remote_sha256.json">全部原件SHA清单</a></p>'
    (OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>高层未来事件技能开发</title><style>body{font:16px Microsoft YaHei,sans-serif;margin:32px;max-width:1450px}img{width:100%}td,th{border:1px solid #ccc;padding:7px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'+body+'</html>',encoding='utf-8')
    print(json.dumps(dict(offline_states=10,offline_arms=20,planning_terminal=(run/'results.json').exists(),report=str(OUT/'index.html')),ensure_ascii=False))


if __name__=='__main__':main()
