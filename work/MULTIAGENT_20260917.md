# RiskWeaver 多大模型协作实验

使用 `sim.llm.multiagent.mode` 选择 `single`、`conditional_critic`、`parallel_arbiter` 或 `parallel_memory`。默认 `single` 不创建协作控制器，沿用既有画像排序或自由规划路径。连续轨迹仍由 diffusion 生成，既有语义、动力学、道路和背景安全验证继续决定是否执行。

## 实现与开源参考

本实现参考以下公开代码的控制流设计，使用 Python 标准库实现有限状态流程，未复制第三方源代码，也未添加框架依赖：

- [LangGraph 官方工作流代码](https://docs.langchain.com/oss/python/langgraph/workflows-agents)：并行分支再聚合、条件路由、评价后修正。对应两个隔离提案工作器、确定性 Borda 仲裁以及条件触发的单次质疑。
- [AutoGen 终止条件实现](https://github.com/microsoft/autogen/blob/main/python/packages/autogen-agentchat/src/autogen_agentchat/conditions/_terminations.py)：显式限制协作轮数和终止条件。当前每次在线规划最多两次 API 请求，单次请求设置超时并关闭 SDK 重试。

按需质疑的触发证据包括无有效历史、多个候选、首个计划无效。服务失败直接传播，不能通过另一个提案掩盖额度耗尽。并行组使用风险与可行性两个不同角色，同一模型固定候选池；它们不意味着独立训练的模型，也不保证建议多样性。

## 离线记忆

`parallel_memory` 需要画像上下文和 `sim.llm.multiagent.memory_path`。`policies/multiagent_memory.py` 加载完整历史尝试，通过独立 LLM 请求生成引用真实记录编号的经验摘要。保留原始 D、执行帧数、失败原因和源场景，摘要只作为假设。创建知识库时排除评估场景，在线再次检查当前场景是否与源场景重叠；发现重叠即报错。无画像上下文时明确拒绝第四模式。

当前 API 冒烟知识库来自历史 23_5.pkl，评估输入来自 6_5.pkl。不能直接把该知识库用于包含 23_5.pkl 的正式评估清单；正式实验须重新划分独立历史集合。

## 已验证结果和边界

2026-09-17 四组单场景闭环冒烟均接受一条计划，但均被 `profile_joint_dynamics` 拒绝，执行攻击帧均为零、D 均为空。LLM 总耗时分别约 4.10、10.01、4.97、4.57 秒，请求数为 1、2、2、2。这是单次测量，不能推断统计显著性或场景生成收益。

独立样本诊断显示攻击开始时 7 个可选联合样本均不可行，jerk 是共同失败原因。提高动力学惩罚系数到 10 的独立诊断也失败；没有放宽硬门槛。正式实验尚未启动。

新增文件职责：`multiagent_planner.py` 编排和调用计量；`multiagent_memory.py` 离线复盘与冻结；`probe_multiagent_api.py` 真实协议冒烟；`launch_multiagent_matrix.py` 四组闭环诊断；`summarize_multiagent.py` 汇总全部尝试。基线原追踪缺少令牌明细，闭环汇总中的基线调用数是逻辑请求数；不能把它当精确计费量。API 协议冒烟独立记录 SDK usage。
