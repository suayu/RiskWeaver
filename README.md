# RiskWeaver（险境织构）

面向自动驾驶测试的危险场景生成与评估研究项目。RiskWeaver在Scenario Dreamer仿真基础上，引入高级LLM攻击规划、Skills场景理解、动态参与者攻击／静态障碍物后备、事件RAG及统一风险评估。

> 本仓库为独立私有研究仓库，包含截至2026-11-06的当前代码快照，不是仅有适配器的代码片段。当前方法仍在验证中，尚未达到“90%以上场景有效危险构建”的验收目标；不可把开发样本、模型服务故障或JSON接收视为总体攻击成功。

## 代码结构

|目录|内容|
|---|---|
|`policies/`|LLM规划器、Skills加载与校验、画像、RAG、动态／静态候选、风险与执行安全门控|
|`skills/`|Skills规则、manifest及版本约束|
|`scenario_generator.py`、`simulator.py`|锚点引导生成、闭环仿真、事件调度与规划观察|
|`models/`、`nn_modules/`、`datasets/`、`datamodules/`|上游生成与行为模型、数据加载和训练实现|
|`experiments/`|多方法对照运行器、统一执行trace和基线接口|
|`work/`|保留的研究原型、分阶段运行入口、回归及只读审计；包括真实使用的嵌套调用链|
|`work/dual_track_20261105/`|最新双轨评估入口、故障续跑、配对统计与攻击归属审计|
|`safe-sim/`|SAFE-SIM相关源代码及原有许可证|
|`cfgs/`、`scripts/`、`tests/`|配置、模型服务启动脚本和回归测试|
|`docs/`|实现、协议、研究限制与阶段记录|
|`README_UPSTREAM.md`|原Scenario Dreamer完整说明和引用|

## 核心流程

1. 从外部冻结场景清单加载道路、自车和周边参与者状态。
2. 高级规划器读取当前事实及候选：Skills开启时使用规则、事实和时序约束；关闭时保留原LLM调用路径。
3. 优先规划其他参与者与自车未来路径形成交互；不适合动态攻击时使用静态组合后备。原物理、道路与背景安全检查仍执行。
4. 扩散模型细化动态轨迹，执行前作约束检查；静态对象按已通过验证的计划创建。
5. 记录原始请求回复、计划、生成／投影／执行轨迹及视频，独立计算攻击归属、风险和驾驶结果。
6. 启用RAG时按当前事实检索事件案例，并只将真实观察闭合为后续知识。RAG与固定规则Skills的职责分开。

## 环境与外部资源

基础依赖见`environment.yml`；CommonRoad-Reach使用独立环境，相关锁定要求见`requirements-commonroad-reach.lock.txt`和`docs/COMMONROAD_REACH_INTEGRATION_20261013.md`，不要直接覆盖已有仿真环境。

```bash
git clone git@github.com:suayu/RiskWeaver.git
cd RiskWeaver
git submodule update --init --recursive
conda env create -f environment.yml
conda activate scenario-dreamer
```

模型权重、Waymo／nuPlan数据、冻结场景pickle、CARLA发行包和实验输出不是源代码，需按`README_UPSTREAM.md`及相应数据／模型授权另外获取。STRIVE为可选外部依赖，通过适配器配置其仓库、独立Python环境和checkpoint。不能把当前Waymo适配称为原论文完整复现。SAFE-SIM实际扩散执行与CtRL-Sim代理评估未来需分别报告。

**路径限制：**研究代码中部分`work/`入口仍冻结了`/home2/zhaoyx/scenario-dreamer`、独立环境或数据路径。新检出目录并不自动迁移服务器运行环境；在不同机器运行前需检查并配置这些路径。该快照保留实际研究代码和历史原型，不声称所有历史入口均可在全新机器上开箱即用。

## 模型服务与凭据

百炼调用包装脚本为`scripts/run_with_llm_provider.sh`。真实密钥只放在未跟踪的环境文件或环境变量中，不能提交到GitHub。

```bash
export DASHSCOPE_API_KEY='<your-own-key>'
# 其他变量名称与加载顺序请按启动脚本和规划器代码检查配置。
```

每场物理SDK预算包括失败回退和JSON修复。明确404／免费额度耗尽可停用模型；普通403／429不能未经证据推断为永久耗尽。2026-11-06用户已确认本轮`kimi-k3.1-pro`免费额度耗尽，后续实验已暂停，未经用户恢复不要继续请求或付费回退。

## 开关和实验框架

基础开关是`sim.llm.skills.enabled`。不同阶段入口还显式控制候选扩展、时序证据、动态优先／静态后备和RAG；正式比较必须使用同一冻结版本、场景、模型、策略和预算，不应只改一个基础布尔值就声称所有原型组合完全相同。

最新配对入口：`work/dual_track_20261105/run_dual_worker_v4.py`；队列：`run_pair.py`。它们通过协议清单运行Skills关闭／开启各20场，每场最多4次实际SDK请求、自然终止或400帧；IDM、seed42、画像／RAG关闭。协议中的场景文件路径与内容SHA需要使用外部数据重新核对；仓库不携带本机实验产物或私有测试数据。

```bash
# 先准备并核对外部清单、模型和路径；prepare-only不调用SDK或执行场景。
python work/dual_track_20261105/run_pair.py --protocol /path/to/verified/protocol
```

这是有外部前置条件的正式运行命令，不应在未核对数据、免费保护与预算时直接执行。单场运行／服务故障继续下一固定场景；效果统计按用户要求排除系统故障，另保留预定、尝试、故障和未尝试台账。正常无攻击、无安全候选等方法结果仍纳入统计，不得为提高覆盖率删除。

## 双轨评价与有效性

- 场景挑战度`I`：有限参考控制库的可行空间损失、最小努力和扰动敏感性，不含实际碰撞或路线偏离标签。
- 后验结果`R`：碰撞、偏离、完成和进度单独汇总。
- 能力分：安全完成为`100I`，否则`100I × 0.5 × progress`，只折半一次。

当前I采用有限控制库、路线走廊和未标定权重，在线值条件化于实际到达的状态；它不是策略无关的完整物理可达性证明，也不是已验证的通用驾驶能力排名。无效或覆盖不足保持null。验证工具见`audit_effect_campaign_v2.py`、`effect_summary.py`、`audit_chat_skill_payloads.py`。旧自由规划缺失request_id时仅能按唯一的同场景、同源帧SDK和计划证据做保守只读关联，不能改写原始trace。

## 代码与权利来源

原Scenario Dreamer论文和依赖的引用保留在`README_UPSTREAM.md`。`SOURCE_PROVENANCE.json`保存此快照的来源分支与基础commit；当前未提交修改也包含在快照中。独立仓库以新的源码快照开始，不复制可能带有凭据或大产物的旧Git历史。

未确认Scenario Dreamer的完整再分发许可前，本仓库保持私有；不擅自给上游代码添加MIT／Apache等新许可证。第三方原有许可证按其文件保留，详见`THIRD_PARTY_NOTICES.md`。私有仓库不代表解决所有第三方授权问题。

## 数据与安全边界

不包含真实`.env`、API密钥、私钥、数据集、权重、虚拟环境、原始SDK日志、实验视频或完整输出目录。代码快照清单和校验信息用于审阅完整性，不表示所有研究性能要求已经达成。自动实验与研究目标保持暂停。

## 启动脚本整理

已删除无剩余代码/测试调用的历史一次性launch_启动器，保留通用入口、当前配对入口和仍被导入的辅助模块。历史日期协议不代表当前可直接运行。具体保留清单与恢复方式见[启动文件清理记录](docs/LAUNCHER_CLEANUP_20261106.md)。
