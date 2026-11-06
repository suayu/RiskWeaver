# Skills与攻击事件RAG实现说明（2026-10-30）

代码已接入远端RiskWeaver-dev工作区，未提交GitHub、未安装依赖或修改Python环境。新两组真实实验已排队，尚不能声称RAG提高了攻击效果。完整skills研究目标仍未完成。

## 1. 分工

Skills保存长期固定的推理规则，RAG只保存与条件相关的历史证据。

|部分|内容|装配方式|
|---|---|---|
|固定skills|角色与身份、当前接近/远离、数值比较、未来时刻与有效期、滚动意图、历史证据使用规范|新清单固定加载，真实请求边界再次确认；两组相同|
|RAG事件库|攻击事件、拒绝与失败、实际执行与未执行、策略条件化观测结果|每组每轮新JSON文件，实验内跨场景持续追加|
|当前事实|当前位姿速度、相对位置与运动、道路/路线、候选策略与强度u|确定性构造查询，不请求LLM自由改写|
|执行与评估|实际攻击帧、EA/D有效覆盖、TTC、可达面积、背景安全|继续使用原执行器和指标，不由历史案例改写门槛|

Skills是提示规则与确定性校验的组合，不等于已经通过训练使模型永远正确。案例中的旧模型意见是不可信历史文本，不具有规则优先级。

## 2. 每次攻击的流程

1. 实验入口创建新库；启用继承时，将指定JSON/旧画像JSONL复制到新文件，保留原始全部行、源SHA和继承数量。源文件不变。
2. 回合开始，记录策略身份、实现文件/检查点指纹及指标协议。不同策略、检查点或协议的新案例不能混成可迁移收益。
3. 发起规划前创建事件。候选产生后，用当前自车速度、车辆前后左右关系、径向变化、道路形态/密度、可用策略和u构造查询。
4. 无RAG组不调用检索器，也不向模型发送案例；两侧都移除原收益表的直接提示注入。画像数值统计仍由原画像模块负责。
5. 有RAG组仅检索本轮知识库：先身份/协议过滤，再按结构条件、BM25标签和连续数值相似度排序。默认最多4条、12000字符；有相关失败时保留反例。相似度不是攻击成功概率。
6. 固定规则和检索证据分别进入真实请求。已有紧凑提示装饰器不能把RAG字段丢掉。原响应schema、候选身份/时域/语义验证、物理SDK预算保持原样。
7. 保存模型计划与完整排名理由、技能版本、接收状态和拒绝原因。执行帧必须同时匹配request_id、target_id及实际执行标记；接收计划不能冒充已经执行。
8. 窗口结束时追加闭合事件。E1按公共逐帧有效记录加权D；普通仿真按原有效窗口记录。无效D保持null。失败、零执行、无攻击事件和未知测量均保留。
9. 下一次规划才可检索已经闭合的旧事件。当前未完成事件和同回合当前/未来帧不进入检索。

真实请求边界会记录`rag_actual_wire`；逐次查询写入`rag_queries.jsonl`。普通仿真和E1结果增加`rag`汇总，知识库落盘错误显式标为运行故障，不当正常无攻击。突然被外部终止的未完成事件不能伪造结算或有效D，应依原始trace另行审计。

## 3. 策略条件化证据的含义

每条新事件包含：画像版本、场景条件、策略/目标/锚点、u、规划意见、接收与实际执行帧、D及有效帧、指标协议、EA/TTC、可达面积变化比例、背景安全、生成/运行失败原因。

`policy_conditioned_evidence`表示“在此自车策略与协议下观察到的结果”，不是与无攻击对照相比的因果收益。只有实际暴露、有效测量、背景安全且无该事件失败时，才标为可用于条件化收益参考。所有不符合者仍留在库里。

旧`strategy_returns.jsonl`可显式继承；原字段完整保留在`legacy_original`。缺少检查点或指标协议的旧行只能作为同policy_id的低可信案例，不能伪装成已验证可迁移收益。旧行缺有效观测帧时D按未知处理，原D仍在原件字段中保留。RAG模式不会隐式读取旧`ego_profile.history_paths`；原自由规划路径仍保留其原行为。

## 4. 主要代码

远端根目录：`/home2/zhaoyx/scenario-dreamer`。

|文件|修改|
|---|---|
|`policies/rag_store.py`|新库、显式继承、完整旧表升级、源SHA、锁、原子写入、幂等和D语义|
|`policies/rag_retrieval.py`|当前事实相关性、策略/协议隔离、反例、上下文预算与审计|
|`policies/rag_runtime.py`|实验/回合/事件生命周期、真实请求桥、实际执行关联及观测写回|
|`policies/ego_profile.py`|升级时禁用旧案例直塞与隐式旧库读取；保留原审计JSONL和画像统计|
|`policies/skills_planner.py`、`skill_registry.py`|当前事实检索入口、固定规则加载；兼容旧冻结技能目录|
|`scenario_generator.py`|规划前建账并绑定最终运行时工厂，接收与窗口结算|
|`run_simulation.py`、`experiments/comparison/run_e1_group.py`|每轮新库、每场归属、执行观测、失败与最终汇总|
|`cfgs/sim/base.yaml`|RAG和继承参数|
|`skills/riskweaver/rw-current-fact-reasoning/SKILL.md`|固定事实/时间/数字推理规范|
|`skills/riskweaver/rw-case-evidence/SKILL.md`|固定历史案例使用规范|
|`work/rag_integration_20261030`|两组准备/真实运行/审计入口及回归测试|

本地完整修改镜像：`work/rag_integration_20261030/source_after`；原件在`source_before`，远端安装前后有SHA及备份收据。未改扩散模型、权重、8联合样本、inner_lr/inner_beta/n_guide_steps或背景安全门槛。

## 5. 参数与复用

公共配置：

```yaml
sim:
  rag:
    enabled: false          # 是否检索并发送案例
    record_events: true     # 新实验记录事件；两组均开启
    inherit: false          # 新实验是否复制指定旧库
    inherit_path: null      # JSON库或旧strategy_returns.jsonl
    top_k: 4
    max_context_chars: 12000
    min_score: 0.15
    policy_version: null
    policy_checkpoint_path: null
```

RAG开启要求skills开启；画像开关独立。`knowledge_mode=attach`仅供父启动器创建本轮新库后内部绑定同一实验，需匹配实验身份；常规新实验用`new`，不要把attach用于继续写历史原库。

两组准备与启动示例（在远端工作区；名称必须是未存在的新目录）：

```bash
PYTHONPATH=.:safe-sim:safe-sim/trajdata/src \
/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python \
work/rag_integration_20261030/prepare_pair.py \
  --output work/rag_pair_NEW_PROTOCOL \
  --experiment-root experiments/rag_pair_NEW_RESULTS \
  --scene-indices 74 182 --profile off

# 需要继承时，在prepare_pair命令增加：
# --inherit --knowledge-source /绝对路径/attack_knowledge_OLD.json

PYTHONPATH=.:safe-sim:safe-sim/trajdata/src \
/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python \
work/rag_integration_20261030/run_pair.py --protocol work/rag_pair_NEW_PROTOCOL
```

准备阶段不发API、不执行场景，冻结输入、源码和两组命令。继承时先冻结同一个种子副本，再给两组分别复制，后续各自写回，不能互读本轮另一组的知识。

原始开发十场也可用`--scene-indices 53 56 74 80 103 140 181 182 207 220`，但当前只启动两场×两组的小规模验证，不自动扩展。`--profile on`可共同开启画像，不能把画像差异当作本轮RAG自变量。

## 6. 验证与当前状态

- 新库/源文件只读继承/完整旧表/多进程/幂等/null语义、检索过滤/真实执行归属/SDK边界/实际时序工厂/两组配置已覆盖。
- 分批回归：20项存储/检索/生命周期、4项两组配置与资源边界、76项原skills/执行兼容与真实时序链、9项候选grounding，共109项；再加入原画像39项后，最终一条联合命令148项全部通过。真实链用SDK替身，不冒充真实模型效果。日志与JUnit在work/rag_analysis_20261030/final_regression.*。
- 13个公共文件与新增技能已安装并核对，旧工作区更改保留；新库JSON无新第三方依赖。Hydra有原有`_self_`配置提醒，未改与任务无关的默认组合顺序。
- `skills_rag_two_arm_development2_20261030_r1`在CUDA模型加载前因显存准入失败：0场景执行、0API；原件保留。
- 新任务`skills_rag_two_arm_development2_20261030_r2`，协议`work/skills_rag_two_arm_preparation_20261030_r3`，373源码/配置，4预定尝试，独立空库，已排队。排队准入18GiB（含1GiB初始化余量），运行时自己14GiB上限+3GiB预留；只跑一个自己的工作进程，不降低围栏、不迁移/重启已有任务。
- 成效尚未验证：需核对实际SDK中检索内容、真实写回、无跨组泄漏、上层接收/事实正确性、预算与成本。是否提高挑战性需后续完整配对结果，当前不能下结论。

两组终态后：

```bash
PYTHONPATH=. /home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python \
work/rag_integration_20261030/audit_pair.py \
experiments/skills_rag_two_arm_development2_20261030_r2
```

保留全部预定分母、失败和未尝试。高层接收、实际执行、文字正确与实测威胁分别报告；接受JSON或检索命中不代表全部推理正确。

## 7. 后续科研验证

先完成本轮真实闭环和案例实际入模审计，再决定是否扩展开发十场/原留出。RAG只新增这一个自变量，不同时启用尚未装配的扩散修正或提示压缩原型。继续保留原时序/画像/静态/NL/可控性、真实性与成本等未完成项目；本次代码集成不能替代完整skills目标验收。
