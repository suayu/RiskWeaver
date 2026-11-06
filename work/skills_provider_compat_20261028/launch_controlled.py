"""复用原真实排队循环和四组分母；追加公共传输修复的验收，不自动重试。"""
from pathlib import Path
import sys

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0, str(ROOT/'work/skills_four_arm_launch_20261026'))
import launch as previous
from audit_controlled import audit_controlled


def main():
    original = previous.previous.audit_four_arm
    previous.previous.audit_four_arm = audit_controlled
    try:
        return previous.main()
    finally:
        previous.previous.audit_four_arm = original


if __name__ == '__main__':
    main()
