"""将离线检索身份与画像候选输入解耦，并校验训练/评估模式。"""
from pathlib import Path
root=Path('/home2/zhaoyx/scenario-dreamer')

def change(name, pairs):
    path=root/name
    source=path.read_text(encoding='utf-8')
    for old,new in pairs:
        assert source.count(old)==1,(name,source.count(old),old[:80])
        source=source.replace(old,new)
    path.write_text(source,encoding='utf-8')

change('policies/ego_profile.py',[(
'''class ProfileAttackPipeline:
    def __init__(self, sim_cfg):
        from types import SimpleNamespace
        settings = getattr(sim_cfg,'ego_profile',SimpleNamespace())
        identity = str(getattr(settings,'policy_id','') or '|'.join(str(getattr(sim_cfg,k,'')) for k in ('policy','rl_model_path','rl_model_name')))
        self.policy_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
''',
'''def policy_id_for_sim(sim_cfg):
    """在画像开关两侧使用同一自车策略身份，避免跨策略检索。"""
    settings = getattr(sim_cfg,'ego_profile',None)
    identity = str(getattr(settings,'policy_id','') or '|'.join(str(getattr(sim_cfg,k,'')) for k in ('policy','rl_model_path','rl_model_name')))
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


class ProfileAttackPipeline:
    def __init__(self, sim_cfg):
        from types import SimpleNamespace
        settings = getattr(sim_cfg,'ego_profile',SimpleNamespace())
        self.policy_id = policy_id_for_sim(sim_cfg)
''')])

change('policies/multiagent_planner.py',[(
'''    def retrieve(self, context):
        """仅检索同画像策略身份的历史，优先场景条件精确匹配，不在线写回。"""
        if self.memory is None:
            return []
''',
'''    def retrieve(self, context, planning_regime='profile'):
        """只检索同一规划模式和策略身份的历史，避免数据泄漏。"""
        if self.memory is None:
            return []
        # 旧库只来自画像模式；无画像组必须使用单独冻结的训练库。
        bank_regime = self.memory.get('planning_regime', 'profile')
        if bank_regime != planning_regime:
            raise ValueError('offline memory planning regime mismatch')
'''),(
'''    def run(self, planner, env_state, instruction, planner_kwargs=None, context=None, scene_image=None):
''',
'''    def run(self, planner, env_state, instruction, planner_kwargs=None, context=None, scene_image=None,
            memory_context=None):
'''),(
'''        if self.mode == 'parallel_memory' and context is None:
            raise ValueError('parallel_memory requires scene-conditioned profile context')
''',
'''        if self.mode == 'parallel_memory' and context is None and memory_context is None:
            raise ValueError('parallel_memory requires scene-conditioned memory context')
'''),(
'''        memory = self.retrieve(context)
        catalog_hash = fingerprint((context or {}).get('candidates', []))
''',
'''        memory = self.retrieve(memory_context if memory_context is not None else context,
                               'profile' if context is not None else 'unprofiled')
        catalog_hash = fingerprint((context or {}).get('candidates', env_state.get('agents', [])))
''')])

change('scenario_generator.py',[(
'''            else:
                attack_plan = self.multiagent_planner.run(self.llm_planner, env_state, selected_instruction,
                                                         planner_kwargs=planner_kwargs)
        planner_wall_seconds = time.monotonic() - planner_started
''',
'''            else:
                memory_context = None
                if self.multiagent_planner.mode == 'parallel_memory':
                    from policies.ego_profile import policy_id_for_sim, scene_conditions
                    # 无画像时只计算检索身份；不启动画像候选排序或改变原规划器输入。
                    memory_context = dict(policy_id=policy_id_for_sim(self.cfg.sim),
                                          scene_id=str(getattr(env,'current_scene_id','unknown')),
                                          scene_conditions=scene_conditions(env))
                attack_plan = self.multiagent_planner.run(self.llm_planner, env_state, selected_instruction,
                                                         planner_kwargs=planner_kwargs,
                                                         memory_context=memory_context)
        planner_wall_seconds = time.monotonic() - planner_started
''')])
