# 静态路线＋后备物理15米技能入口

应用技能，不是Codex自动技能。默认关闭；保留原动态优先、原静态成功路径、背景安全与原调用预算。没有安装依赖、改公共默认入口、训练或提交GitHub。

`make_static(..., enabled=False)`返回原StaticCombinationPlanner；`runtime.install(False)`不修改构造函数。明确开启时，只替换新建DynamicFirstStaticPlanner的静态分支，动态对象与总预算保留。原朝向检查失败才用路线首次相交≥15米且自车—新物体OBB间距≥15米；原成功分支不被重新收紧。

固定规则在SKILL.md和manifest.json。模型请求得到版本/距离口径/背景裕度字段；目录构建与最终复核使用相同适配器，trace另存几何和最终验证，不把它们当新鲜LLM或执行证据。直接build_context目录预览也使用同一规则。JSON类型适配不改变几何结果。

## 调用

在原工作入口完整参数前增加：

```text
python -m work.skills_static_physical15_20261102.run_worker
  --base-worker /home2/zhaoyx/scenario-dreamer/work/<已冻结工作入口>/run_worker.py
  --static-route-physical15 off|on
  <原有 --output-dir、--skills、模型、预算、清单等完整参数>
```

`on`要求显式`--skills on`，因为当前仅实现Skills静态分支，不得用于原自由LLM基线并谎称双方已应用相同新门槛。最终原机制对照若采用新距离规则，还需要对双方装配共同门控并独立验收。`off`原样透传旧参数；既有输出目录禁止覆盖；输出由公共runner创建，本入口退出时才保存收据。`--prepare-only`不创建结果目录。

## 当前证据

- 同250初态目录审计：原静态201、动静态并集243；严格路线后备静态239、并集250，原可用无退化。仅目录证据。
- 新包11项组件/公共引擎测试通过；包含动态优先、总预算、最新自车状态与背景安全复查、独立预览和异常恢复。另有入口测试，结果以实际JUnit为准。
- `audit_seven_wiring.py`对原91/100/129/211/238/241/246七场做原目录与新静态请求/最终复查。回复明确为DETERMINISTIC_WIRING_FIXTURE_NOT_LLM；不调用API、不env.step、D=null、碰撞未知。首次七场接线完成，后续完整manifest/CLI版本须以其独立新快照再核对。

新鲜LLM多数覆盖、真实运行、出现自然性、实际威胁和完整A—E仍未完成。可见模型不代表免费可调用；本包不会绕过用户仅免费限制或恢复耗尽模型。
