import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
base=Path("/home2/zhaoyx/scenario-dreamer/experiments/riskweaver_rl_profile_paired_calibration_20260923")
s=json.loads((base/"paired_summary.json").read_text())
font="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"; font_manager.fontManager.addfont(font)
plt.rcParams["font.sans-serif"]=[font_manager.FontProperties(fname=font).get_name()]; plt.rcParams["axes.unicode_minus"]=False
rows=s["paired"]["rows"]; names=[Path(r["scenario_source"]).name for r in rows]; x=np.arange(len(rows)); w=.35
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
axes[0].bar(x-w/2,[r["off_attack_frames"] for r in rows],w,label="画像关闭",color="#4C78A8")
axes[0].bar(x+w/2,[r["on_attack_frames"] for r in rows],w,label="画像开启",color="#F58518")
axes[0].set_xticks(x,names); axes[0].set_ylabel("实际攻击执行帧"); axes[0].set_title("画像开启减少自由规划漏攻"); axes[0].legend(); axes[0].grid(axis="y",alpha=.2)
for i,r in enumerate(rows):
 for pos,key in ((i-w/2,"off_scene_d"),(i+w/2,"on_scene_d")):
  v=r[key]
  if v is not None: axes[1].scatter(pos,v,s=70,color="#4C78A8" if "off" in key else "#F58518")
  else: axes[1].text(pos,0.02,"D 无效",ha="center",rotation=90,color="#F58518",fontsize=8)
axes[1].set_xticks(x,names); axes[1].set_ylabel("全场危险度 D"); axes[1].set_title("全场 D 含无攻击时的自然风险"); axes[1].grid(axis="y",alpha=.2)
fig.suptitle("GPUDrive RL 三场景画像配对校准（预注册验收未通过）",fontsize=14)
fig.text(.5,.01,"关闭组 0/3 场攻击；开启组 2/3 场、38 帧攻击；两组服务失败与背景碰撞均为 0。",ha="center",fontsize=9,color="#555555")
for ax in axes: ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(rect=[0,.06,1,.92]); fig.savefig(base/"GPUDrive_RL三场画像配对校准.png",dpi=200)
