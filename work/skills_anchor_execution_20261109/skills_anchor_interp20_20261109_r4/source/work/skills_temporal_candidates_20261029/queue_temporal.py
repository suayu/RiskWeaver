"""复用原单GPU共享队列；新实验不与已终态契约诊断拼接。"""
from pathlib import Path
import sys
ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_rolling_smoke_launch_20261029'))
import queue_smoke as previous
previous.RUN=ROOT/'experiments/skills_temporal_candidates_scene074_20261029_r1'


if __name__=='__main__':
    try:previous.main()
    except Exception as e:
        previous.save(previous.RUN/'launcher_result.json',dict(status='controller_failure',error_type=type(e).__name__,reason=str(e),goal_complete=False))
        raise
