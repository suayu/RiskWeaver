---
name: rw-static-fallback
description: 无可接受动态车辆候选时，选择静态物体组合阻挡自车未来路线。
---

# 动态优先与静态组合兜底

- 调度器先构造并验证动态车辆攻击目录。有可用动态候选时，优先选择车辆攻击。当前有车不等于该车一定具备可行攻击意图。
- 只有动态目录为空时，进入静态组合分支。模型服务错误、输出契约错误和“没有可攻击车辆”是不同原因，不得相互替代。
- 静态目录的每个 ID 对应一组完整 `placements`，包含一到两个已注册模板；每个模板还可展开为多根围挡、多个锥桶或水马。选择整个候选 ID，不自行修改坐标、类型、数量和摆放参数。
- 本分支的动态 `ranking` 为空。仍按原 JSON 契约输出具体 `scene_analysis`、`no_attack_reason` 和非空 `obstacle_candidate_id`；其中 no_attack_reason 应说明没有采用动态攻击的原因，不要把已选静态攻击描述为完全没有攻击。
- 优先考虑确实侵入自车未来路线的障碍布局。静态存在不等于必然碰撞；位置、影响时刻、剩余空间和背景交通证据都应说明来源。
- 几何未占用面积大于零仅代表局部道路走廊里仍有未被静态物体占据的面积，不保证自车可躲避，也不是动态可达面积或危险度 D。保持原背景安全和障碍物边界校验。
- 若目录本身为空，如实报告道路几何、剩余距离、已有物体或背景交通造成的限制。禁止凭空编造可接受方案或把服务失败算作攻击成功。

## Dynamic suitability and fallback
Prefer a feasible dynamic opportunity with supported future interaction. A nonempty dynamic catalog alone is not proof of a threatening attack. If the dynamic branch is unavailable or declines and a legal static combination is published, select the static fallback within the shared request budget. Service failures remain technical failures. Do not fabricate objects or claim placement, execution or collision before observing it.
