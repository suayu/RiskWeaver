"""只扩展已开启协作的上下文，供运行时检查场景泄漏。"""
from pathlib import Path
import time

root = Path('/home2/zhaoyx/scenario-dreamer')
path = root/'scenario_generator.py'
raw = path.read_bytes()
old = '                context["policy_id"] = self.profile_pipeline.policy_id\n'
new = old + '                context["scene_id"] = self.profile_pipeline.pending["scene_id"]\n'
text = raw.decode('utf-8')
if text.count(old) != 1:
    raise RuntimeError('context insertion anchor mismatch')
(root/'work'/('generator_before_memory_guard_'+str(time.time_ns())+'.py')).write_bytes(raw)
path.write_text(text.replace(old, new), encoding='utf-8')
