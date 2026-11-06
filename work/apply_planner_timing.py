"""在已检查的规划调用边界增加只读耗时追踪。"""
from pathlib import Path
path=Path('/home2/zhaoyx/scenario-dreamer/scenario_generator.py')
source=path.read_text(encoding='utf-8')
old='''        if self.profile_enabled:
            from policies.profile_planner import rank_profile_candidates
            context = self.profile_pipeline.context(env, self._iterative_stage)
'''
new='''        # 在现有 LLM 调用边界计时；单 Agent 与协作模式使用相同墙钟口径。
        import time
        planner_started = time.monotonic()
        if self.profile_enabled:
            from policies.profile_planner import rank_profile_candidates
            context = self.profile_pipeline.context(env, self._iterative_stage)
'''
assert source.count(old)==1
source=source.replace(old,new)
old='''                attack_plan = self.multiagent_planner.run(self.llm_planner, env_state, selected_instruction,
                                                         planner_kwargs=planner_kwargs)
        if query_trace is not None:
'''
new='''                attack_plan = self.multiagent_planner.run(self.llm_planner, env_state, selected_instruction,
                                                         planner_kwargs=planner_kwargs)
        planner_wall_seconds = time.monotonic() - planner_started
        if query_trace is not None:
'''
assert source.count(old)==1
source=source.replace(old,new)
old="""                                'planner_trace': getattr(self.llm_planner, 'last_trace', None)})
"""
new="""                                'planner_trace': getattr(self.llm_planner, 'last_trace', None),
                                'planner_wall_seconds': planner_wall_seconds})
"""
assert source.count(old)==1
source=source.replace(old,new)
path.write_text(source,encoding='utf-8')
