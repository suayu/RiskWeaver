"""使用精确文本锚点增量接入，先保存修改前快照，保留既有未提交改动。"""
from pathlib import Path
import hashlib
import json
import time

root = Path('/home2/zhaoyx/scenario-dreamer')
backup = root / 'work' / ('multiagent_before_' + str(time.time_ns()))
backup.mkdir(parents=True)
hashes = {}


def edit(name, changes):
    path = root / name
    raw = path.read_bytes()
    hashes[name] = hashlib.sha256(raw).hexdigest()
    target = backup / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    text = raw.decode('utf-8')
    for old, new in changes:
        if text.count(old) != 1:
            raise RuntimeError('修改锚点不唯一: ' + name)
        text = text.replace(old, new)
    path.write_bytes(text.encode('utf-8'))


edit('scenario_generator.py', [
    ('        self.profile_pipeline = None\n',
     '        # 默认单 Agent 不创建协作控制器，完整保留既有调用分支。\n'
     '        self.multiagent_planner = None\n'
     '        multi_cfg = getattr(getattr(self.cfg.sim, "llm", None), "multiagent", None)\n'
     '        if str(getattr(multi_cfg, "mode", "single")) != "single":\n'
     '            from policies.multiagent_planner import MultiAgentPlanner\n'
     '            self.multiagent_planner = MultiAgentPlanner(multi_cfg)\n'
     '        self.profile_pipeline = None\n'),
    ('            attack_plan = rank_profile_candidates(self.llm_planner,env_state,selected_instruction,context,scene_image)\n',
     '            if getattr(self, "multiagent_planner", None) is None:\n'
     '                attack_plan = rank_profile_candidates(self.llm_planner,env_state,selected_instruction,context,scene_image)\n'
     '            else:\n'
     '                # 画像身份和场景条件仅扩展协作输入，不改变原单 Agent 提示。\n'
     '                context["policy_id"] = self.profile_pipeline.policy_id\n'
     '                context["scene_conditions"] = self.profile_pipeline.pending["scene_conditions"]\n'
     '                attack_plan = self.multiagent_planner.run(self.llm_planner, env_state, selected_instruction,\n'
     '                                                         context=context, scene_image=scene_image)\n'),
    ('            attack_plan = self.llm_planner.generate_attack_plan(env_state, selected_instruction, **planner_kwargs)\n',
     '            if getattr(self, "multiagent_planner", None) is None:\n'
     '                attack_plan = self.llm_planner.generate_attack_plan(env_state, selected_instruction, **planner_kwargs)\n'
     '            else:\n'
     '                attack_plan = self.multiagent_planner.run(self.llm_planner, env_state, selected_instruction,\n'
     '                                                         planner_kwargs=planner_kwargs)\n'),
])
edit('cfgs/sim/base.yaml', [
    ('llm:\n', 'llm:\n'
     '  # 四组消融只切换此参数；single 完整沿用当前调用流程。\n'
     '  multiagent:\n'
     '    mode: single # single / conditional_critic / parallel_arbiter / parallel_memory\n'
     '    timeout_seconds: 60\n'
     '    memory_path: null # 仅 parallel_memory 读取冻结的离线复盘知识库\n'),
])
(backup / 'hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
print(str(backup))
