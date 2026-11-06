"""只增加拒绝时审计，保持原检查顺序与拒绝条件。"""
from pathlib import Path
import time

root=Path('/home2/zhaoyx/scenario-dreamer')
path=root/'run_simulation.py'
raw=path.read_bytes()
text=raw.decode()
changes=[
    ('            try:\n                self.env.prepare_background_traffic()\n',
     '            raw_joint_for_audit = None\n            try:\n                self.env.prepare_background_traffic()\n                raw_joint_for_audit = self.env.pending_joint_trajectory\n'),
    ('            except NoSafeJointCandidate as error:\n',
     '            except NoSafeJointCandidate as error:\n                from policies.rejection_audit import rejection_record\n                trace.append(rejection_record(self.env, profile_pipeline, str(error), raw_joint_for_audit))\n'),
]
for old,new in changes:
    if text.count(old)!=1:
        raise RuntimeError('audit insertion anchor mismatch')
    text=text.replace(old,new)
(root/'work'/('run_before_rejection_audit_'+str(time.time_ns())+'.py')).write_bytes(raw)
path.write_text(text,encoding='utf-8')
