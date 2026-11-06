import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
base=Path("/home2/zhaoyx/scenario-dreamer/experiments/riskweaver_rl_profile_smoke_20260923")
d=json.loads(next(base.rglob("attempt_result.json")).read_text())
font="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font_manager.fontManager.addfont(font)
plt.rcParams["font.sans-serif"]=[font_manager.FontProperties(fname=font).get_name()]
plt.rcParams["axes.unicode_minus"]=False
items=[("危险度 D",d["scenario_danger_score"],0.12),
       ("实际攻击帧",d["attack_executed_frames"],25),
       ("平均最小 TTC（秒）",4.1549584256203875,6),
       ("能力分 S/100",d["autonomous_driving_ability_score"],1)]
fig,axes=plt.subplots(1,4,figsize=(12,3.4))
colors=["#E45756","#F58518","#4C78A8","#72B7B2"]
for ax,(label,value,limit),color in zip(axes,items,colors):
 ax.bar([label],[value],color=color,width=.55)
 ax.set_ylim(0,limit)
 ax.text(0,value+limit*.03,f"{value:.3f}" if isinstance(value,float) else str(value),ha="center",fontsize=10)
 ax.grid(axis="y",alpha=.2)
 ax.spines[["top","right"]].set_visible(False)
 ax.tick_params(axis="x",labelrotation=10)
fig.suptitle("GPUDrive RL 自车画像攻击闭环冒烟（验收通过）",fontsize=14)
fig.text(.5,.015,"2 个攻击计划、18 个执行帧、背景碰撞 0；第 30 帧后续投影求解失败被安全拒绝。",ha="center",fontsize=9,color="#555555")
fig.tight_layout(rect=[0,.07,1,.92])
fig.savefig(base/"GPUDrive_RL画像攻击冒烟摘要.png",dpi=200)
