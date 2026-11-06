"""复用已批准的GPU0/14GiB围栏，仅更换新实验目录，不修改旧排队器。"""
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_rolling_smoke_launch_20261029'))
import queue_smoke as previous
previous.RUN=ROOT/'experiments/skills_rolling_contract_scene074_20261029_r1'


if __name__=='__main__':
    try:previous.main()
    except Exception as error:
        previous.save(previous.RUN/'launcher_result.json',dict(status='controller_failure',
            error_type=type(error).__name__,reason=str(error),goal_complete=False))
        raise
