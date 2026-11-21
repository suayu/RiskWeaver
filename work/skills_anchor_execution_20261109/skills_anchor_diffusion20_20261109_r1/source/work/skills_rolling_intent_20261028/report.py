"""生成中文时序诊断及英文科研备用图；明确区分历史审计、新实验和未验证原型。"""
from pathlib import Path
import html
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from temporal_audit import sample

OUT=Path('D:/RiskWeaver/outputs/skills_shared_resource_20261028')
DEST=OUT/'future_ego_rolling_intent_20261029_v2'
ZH=['旧机制','目录无知识','有知识无扩展','完整技能组']
ARMS=['legacy_shared_recovery','catalog_no_knowledge','skills_no_expansion','skills_full_shared_recovery']


def main():
    DEST.mkdir(exist_ok=False)
    evidence=json.loads((OUT/'scene074_temporal_intent_audit_20261028.json').read_text(encoding='utf-8'))
    scientific=json.loads((OUT/'terminal_scientific_audit_20261029/scientific_audit.json').read_text(encoding='utf-8'))
    physical=json.loads((OUT/'terminal_physical_realism_20261029.json').read_text(encoding='utf-8'))
    for zh in [True,False]:
        plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'] if zh else ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus']=False
        suffix='zh' if zh else 'en'
        fig,axes=plt.subplots(1,3,figsize=(16,4.8))
        for p,ax in zip(evidence['plans'],axes):
            t=np.arange(31)*.1;ego=np.asarray(p['ego_source']);intent=sample(p['anchors'],t)
            g=np.linalg.norm(intent-(ego[:2]+t[:,None]*ego[2:4]),axis=1)
            ax.plot(t,g,label='锚点意图与自车匀速假设' if zh else 'Sparse intent vs ego CV',color='#c85d1e')
            joint=p['joint'];i=joint['agent_ids'].index(p['target_id']);xy=np.asarray(joint['positions_global'][i])
            jt=np.arange(1,len(xy)+1)*.1;jg=np.linalg.norm(xy-(ego[:2]+jt[:,None]*ego[2:4]),axis=1)
            ax.plot(jt,jg,label='投影后扩散预测与自车匀速假设' if zh else 'Post-projection prediction vs ego CV',color='#177e89')
            ax.axvspan(0,1,color='#dcebd8',alpha=.6,label='本次执行有效期' if zh else 'Execution lease')
            k=int(np.argmin(g));ax.scatter([t[k]],[g[k]],color='#c85d1e');ax.annotate(f'{t[k]:.1f}s / {g[k]:.2f}m',(t[k],g[k]),xytext=(8,9),textcoords='offset points')
            ax.set(title=(f"源帧{p['source_step']}，攻击车{p['target_id']}" if zh else f"Source {p['source_step']}, target {p['target_id']}"),
                xlabel='相对预测时间（秒）' if zh else 'Relative forecast time (s)',ylabel='中心距离（米）' if zh else 'Center distance (m)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('历史74号：预测时域不等于实际执行有效期；距离不是碰撞证书' if zh else 'Historical scene 74: forecast horizon is not execution lease; center gap is not collision evidence')
        fig.tight_layout();fig.savefig(DEST/f'timing_{suffix}.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(16,5.5))
        for p,ax in zip(evidence['plans'],axes):
            a=np.asarray(p['anchors']);ego=np.asarray(p['ego_source']);t=np.arange(4.)
            joint=p['joint'];j=joint['agent_ids'].index(p['target_id']);xy=np.asarray(joint['positions_global'][j]);actual=np.asarray([r['position_world'] for r in p['tracking']['actual_declared_lease_samples']])
            ax.plot(a[:,0],a[:,1],'o--',label='LLM选中稀疏意图' if zh else 'LLM-selected sparse intent',color='#c85d1e')
            ax.plot(xy[:,0],xy[:,1],label='投影后扩散预测' if zh else 'Post-projection diffusion',color='#177e89')
            ax.plot(actual[:,0],actual[:,1],lw=3,label='实际攻击车：仅本次有效期' if zh else 'Actual attacker: lease only',color='#7c4d9e')
            e=ego[:2]+t[:,None]*ego[2:4];ax.plot(e[:,0],e[:,1],'s:',label='自车匀速未来假设' if zh else 'Ego future CV hypothesis',color='#356cbd')
            for n,v in enumerate(a):ax.annotate(f'{n}s',v,xytext=(3,4),textcoords='offset points',fontsize=8)
            ax.set(title=(f"源帧{p['source_step']}，车{p['target_id']}" if zh else f"Source {p['source_step']}, target {p['target_id']}"),xlabel='世界坐标X（米）' if zh else 'World X (m)',ylabel='世界坐标Y（米）' if zh else 'World Y (m)')
            ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle('历史轨迹：不把后续重规划后的实际状态归因于已过期计划' if zh else 'Historical paths: later replanned states are not attributed to expired intent')
        fig.tight_layout();fig.savefig(DEST/f'paths_{suffix}.png',dpi=160);plt.close(fig)
        c=evidence['replans'][0];fig,ax=plt.subplots(figsize=(9,4.5));times=c['absolute_times_seconds']
        ax.plot(times,c['raw_overlap_deviations_m'],'o-',label='原始同时间偏差' if zh else 'Raw same-time deviation')
        ax.plot(times,c['tracking_aligned_shape_deviations_m'],'s-',label='扣除起点跟踪偏差后的形状变化' if zh else 'Shape change after source-offset alignment')
        ax.set(title='源帧0→11：同一攻击车的剩余未来意图发生跳变' if zh else 'Source 0 to 11: remaining intention changes for the same attacker',
            xlabel='场景绝对时间（秒）' if zh else 'Absolute scene time (s)',ylabel='位置偏差（米）' if zh else 'Position deviation (m)');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(DEST/f'continuity_{suffix}.png',dpi=160);plt.close(fig)
    lines=['# 自车未来参照、同次攻击连续性与扩散锚点诊断','',
        '本报告分开使用两套证据：锚点诊断来自已归档的20261026_r2完整技能组74号历史场景；四组性能表来自刚完成的20261028_r2共享资源实验，不拼接两轮结果。',
        '', '## 1. 74号实际发现', '',
        '- 4个锚点对应0/1/2/3秒；一次执行有效期10帧=1秒。源帧0与11使用同一攻击车0；源帧55换成车1，且车0已失活，是新攻击。',
        '- 源帧0意图与自车CV假设最近时刻1.4秒、中心距离2.705米，落在1秒执行有效期之外。源帧11最近时刻0.4秒、3.005米，落在有效期内。末端锚点可能是交互后的驶离位置，不能把3秒终点当成唯一碰撞时刻。',
        '- 最近时间和距离沿既有0.1秒网格报告，不声称连续时间精确最小值。新版active-window配方已使用自车CV未来位置；不能概括为全部瞄准自车当前位置。CV不能反映路线转弯、制动和实际策略反应。',
        '- 源帧0→11按同一绝对时刻比较：起点实际跟踪误差2.474米；最大原始未来偏差20.962米；平移扣除跟踪偏差后最大形状变化18.603米。旧执行有效期已结束，这反映预测意图不连续，不是旧控制仍在执行。',
        '', '## 2. 扩散跟随：证据与边界', '',
        '|源帧|预测1/2/3秒锚点位置误差（米）|实际1秒末端误差（米）|',
        '|---|---|---|']
    for p in evidence['plans']:
        tr=p['tracking'];errors='/'.join(f"{r['error_m']:.3f}" for r in tr['predicted_anchor_errors']);lines.append(f"|{p['source_step']}|{errors}|{tr['actual_lease_end_error_m']:.3f}|")
    lines += ['', '预测数组第0项是t+dt，所以1/2/3秒取索引9/19/29。实际误差只比较该计划有效期内的10帧。源帧0/55动力学投影最大位移分别约0.00361/0.00128米；源帧11因历史攻击车豁免未投影。因此本例9–12米远期跟随误差不能主要归因于这次QP投影。没有完整投影前候选/逐项梯度记录，尚不能确定去噪指导、候选选择、动作空间标定分别贡献多少。',
        '', '## 3. 已实现的独立高级规划原型', '',
        '新代码位于work/skills_rolling_intent_20261028，默认关闭，入口--rolling-intent off|on。未改已完成四组源码、扩散层、指标或安全阈值；未提交GitHub。',
        '',
        '1. 提供带来源和绝对时间的自车未来假设：当前状态匀速CV，以及几何覆盖且方向一致时的路线恒速假设。路线尾部不足时明确缺失，不冻结终点或外推。它们不是实际自车策略滚动预测，也没有标定概率。',
        '2. 缓存上次被接收的目标、策略、绝对交互时刻和世界坐标稀疏锚点。跨场景、车辆失活、时间倒退或预测过期明确重置；不把接收记忆当执行证据。',
        '3. 同次攻击生成仍在旧预测覆盖内的剩余意图候选：首点固定实测位置，重叠锚点按绝对时间重采样并分离跟踪平移；旧预测以外的尾部明确采用新候选，不冒充旧已知未来。',
        '4. 未来冲突候选使用同时间自车假设的位置生成，保留原3.8m/s²配方加速度上限。所有新增候选仍经过原背景/动力学screen和原语义验证器。目录上限、响应schema、8个联合采样、求解器与4次SDK总预算不变。',
        '5. 技能要求解释保持/改换目标、预测交互时刻是否落在执行有效期内、过去事件是否需要重新评估。连续性目前是候选和理由依据，没有新增强制门槛；尚未保证LLM始终选择连续候选或每次正确解释。',
        '', '旧语义验证仍使用原CV预检，不因路线假设而静默改变其行为。后续若换成多假设时间对齐预检，须独立协议和对照；不能声称此原型已经完成全部未来规划改造。',
        '', '组件和真实调用链SDK替身、独立入口回归27项通过；新增代码尚未真实API/闭环验证，不把测试通过认定为P1/P2完成。',
        '', '## 4. 问题3的后续方案：仅分析，未修改扩散', '',
        '第一步应记录proposal→projection→execution三阶段相同时间/坐标/动作归一化下的误差、每项指导梯度及安全筛选原因，排除时间索引、单位、运动学映射和梯度尺度问题；历史修过的梯度错误不能未经复核当作仍存在。',
        '第二步在同批8个联合候选中，以实际执行有效期内的锚点位置/速度/交互时刻一致性作选择依据，并继续要求联合安全。若意图不可行，报告冲突而不是强行写入航点或提高碰撞率放行。',
        'AnchDrive可借鉴静态驾驶先验与上下文动态锚点初始化、截断去噪和相对锚点的残差预测。它是端到端规划训练架构，不是给现有TBSim checkpoint换个推理参数即可复现；需配套训练或微调，并处理动作空间及噪声分布。',
        'AnchorDrive可参考时间索引锚点一致性与规则联合引导。RiskWeaver已有锚点损失，新增价值需来自时序、尺度和采样初始化的实际改进；不能仅叠加同类损失宣称已解决。不采用其LLM连续控制路径，继续保持LLM离散意图、diffusion连续轨迹的分工。',
        '可进一步试验位置+速度软约束/增广拉格朗日、动作空间锚点条件与局部残差去噪；先固定优化器做权重标定，再独立做优化器消融。联合比较1秒和3秒锚点误差分位数、实际引导帧、TTC/EA/D、背景安全、真实性与计算预算。',
        '', '资料：[AnchDrive](https://arxiv.org/html/2509.20253v2)；[AnchorDrive](https://arxiv.org/html/2603.02542v1)。以上是迁移方案，不是效果承诺。',
        '', '## 5. 刚完成四组开发实验', '',
        '|组别|预定/尝试/实际执行场景|接收计划/实际攻击帧|攻击场景|严格合格|D均值/有效帧覆盖|生成失败|',
        '|---|---|---|---|---|---|---|']
    for arm,name in zip(ARMS,ZH):
        g=scientific['groups'][arm];d=g['D']['mean'];lines.append(f"|{name}|10/{g['attempted_scenes']}/{g['executed_scenes']}|{g['accepted_plans']}/{g['attack_frames']}|{g['actual_attack_scenes']}/10|{g['strict_qualified_scenes']}/10|{d:.6f}/{g['D_frame_coverage']:.1%}|{g['generation_failures']}|")
    lines += ['', '四组均无运行/最终服务/预测服务失败和背景碰撞；完整技能组7/10达到本轮严格开发运行门槛，旧组4/10。严格合格要求完整终止、真实响应、实际引导攻击、有效D、该场无生成/运行/服务/背景失败及产物齐全。它不是语义正确性或威胁证书。全部组自车碰撞均为0；当前数据不能声称已提升碰撞率。',
        '完整技能组期间qwen3.8-max-0902额度耗尽后自动回退，实际混用模型，不能把整包收益全部归因于skill文本。B/C结果一致，未证明仅增加知识文字有效；C/D还含候选扩展和配方变化。',
        '独立物理真实性审计的全轨迹配对均值差（完整组减旧组，W1越小越接近参考）为：']
    for feature,r in physical['comparisons']['legacy_shared_recovery__vs__skills_full_shared_recovery']['full'].items():
        lines.append(f"- {feature}：{r['mean_right_minus_left']:+.6f}，共同有效{r['paired_valid_scenes']}/10。")
    lines += ['','这些特征均变差，不支持更真实或不劣。旧分箱口径、共同前缀、原始样本口径和全部失败均另存，不挑有利子集。小规模开发与共享GPU时间不能替代独立泛化、控制器因果对照或独占速度实验。',
        '', '## 6. 下一步与未完成项', '',
        '先对新原型做真实74单场诊断，检查原始请求/回复、时刻引用、同次攻击目标与重叠锚点、执行威胁和失败；再冻结版本进行原留出10场旧/新成对验证，保留全部预定分母。此前四组已完成，监控riskweaver已删除，不再轮询终态进程。',
        '最高优先级P1多数正确攻击尚未完成：开发运行门槛通过不等于战术正确；历史自由文字的矛盾与当前未验证项继续保留。P2实际威胁、P3自然语言、P4难度、多策略、画像与静态分支及原方案后续验证仍待完成。',
        '', '## 图与原始资料','', '![时间对齐](timing_zh.png)','', '![未来意图和预测执行](paths_zh.png)','', '![连续性](continuity_zh.png)']
    md='\n'.join(lines)+'\n';(DEST/'诊断与修改说明.md').write_text(md,encoding='utf-8')
    body='<meta charset="utf-8"><style>body{font:16px/1.7 Microsoft YaHei;max-width:1250px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}a{color:#165ba6}</style><h1>自车未来参照与滚动攻击意图</h1>'
    body+='<p>历史74诊断、新四组终态和未验证原型分开报告。扩散仅分析，本轮未修改。</p>'
    for name in ['timing','paths','continuity']:body+=f'<img src="{name}_zh.png">'
    body+='<p><a href="诊断与修改说明.md">完整中文说明</a> · <a href="../terminal_scientific_audit_20261029/index.html">四组逐场统计与全部视频</a> · <a href="../scene074_temporal_intent_audit_20261028.json">74原始重算证据</a></p>'
    body+='<pre>'+html.escape(md)+'</pre>'
    (DEST/'index.html').write_text(body,encoding='utf-8')
    root=OUT/'index.html';prior=root.read_text(encoding='utf-8')
    (DEST/'root_index_before_update.html').write_text(prior,encoding='utf-8')
    root.write_text('<meta charset="utf-8"><p><a href="future_ego_rolling_intent_20261029_v2/index.html">最新：未来自车参照、同次攻击连续性与四组终态（统一0.1秒网格）</a> · <a href="terminal_scientific_audit_20261029/index.html">四组全部视频</a></p>'+prior,encoding='utf-8')
    print(json.dumps(dict(report=str(DEST/'index.html'),figures=6,new_requests=0,new_execution_frames=0)))


if __name__=='__main__':main()
