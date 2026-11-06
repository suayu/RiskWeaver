import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
font_manager.fontManager.addfont(font_path)
cn_font=font_manager.FontProperties(fname=font_path).get_name() 
p=Path('/home2/zhaoyx/scenario-dreamer/experiments/riskweaver_anchor_drive_smoke_20260922')
d=json.loads((p/'before_after_summary.json').read_text(encoding='utf-8'))
plt.rcParams['font.sans-serif']=[cn_font]
plt.rcParams['axes.unicode_minus']=False
labels=['修改前','修改后']
colors=['#8da0cb','#fc8d62']
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
mean=[d[x]['anchor_error_mean_m'] for x in labels]
p95=[d[x]['anchor_error_p95_m'] for x in labels]
x=range(2)
axes[0].bar([i-0.18 for i in x],mean,width=.36,label='平均误差',color='#4c78a8')
axes[0].bar([i+0.18 for i in x],p95,width=.36,label='95分位误差',color='#f58518')
axes[0].set_xticks(list(x),labels); axes[0].set_ylabel('锚点位置误差（米）'); axes[0].set_title('锚点跟随质量'); axes[0].legend()
for i,v in enumerate(mean): axes[0].text(i-.18,v+.08,f'{v:.2f}',ha='center')
for i,v in enumerate(p95): axes[0].text(i+.18,v+.08,f'{v:.2f}',ha='center')
frames=[d[x]['attack_executed_frames'] for x in labels]
axes[1].bar(labels,frames,color=colors,width=.55)
axes[1].set_ylabel('实际攻击执行帧数'); axes[1].set_title('闭环攻击执行覆盖')
for i,v in enumerate(frames): axes[1].text(i,v+.5,str(v),ha='center')
axes[1].text(.5,.93,'两组背景碰撞帧均为 0',transform=axes[1].transAxes,ha='center',bbox=dict(boxstyle='round',fc='#e8f5e9',ec='#43a047'))
fig.suptitle('AnchorDrive 风格锚点引导：单场景探索性前后对照',fontsize=14)
fig.text(.5,.01,'注意：两次百炼规划输出不同，因此该图用于闭环冒烟，不作为严格因果消融。',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=[0,.05,1,.94])
fig.savefig(p/'锚点引导修改前后对照.png',dpi=180)



