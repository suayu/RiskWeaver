"""四组运行元数据修正：复用原启动与验收，仅从冻结清单计算预定尝试数。"""
from pathlib import Path
import json
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0] = [str(ROOT/'work/skills_protocol_preparation_20261025'),
               str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
import launch_four_arm as previous


def install_attempt_count(launcher):
    original = launcher.save
    def save(path, value):
        if path.name == 'launch.json' and isinstance(value, dict) and 'arms' in value:
            # 固定两组的20不是通用分母；失败与未尝试仍属于全部预定尝试。
            manifest = json.loads((path.parent/'manifest.json').read_text())
            indices = manifest['original_scenario_indices']
            if len(indices) != manifest['scenario_count'] or len(set(indices)) != len(indices):
                raise ValueError('planned_scene_count_or_duplicate_mismatch')
            value = dict(value, planned_attempts=len(value['arms'])*len(indices),
                         planned_attempts_basis='frozen_arm_count_times_frozen_scenario_count',
                         planned_scenes_per_arm=len(indices))
        original(path, value)
    launcher.save = save
    return lambda:setattr(launcher, 'save', original)


def main():
    restore = install_attempt_count(previous.launcher)
    try:
        return previous.main()
    finally:
        restore()


if __name__ == '__main__':main()
