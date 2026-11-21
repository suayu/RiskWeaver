"""仅供明确启动时使用；复用空闲GPU排队，四组终态全部审计，不自动重试。"""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

ROOT = Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0, str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
import launch_pool_pair as launcher
import pool_acceptance
from audit_four_arm import audit_four_arm, smoke_prerequisite


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--stage',required=True)
    args,_ = parser.parse_known_args()
    metadata = json.loads((args.protocol/'protocol.json').read_text())
    for item in metadata['required_smoke_receipts']:
        result = smoke_prerequisite(ROOT/item['experiment'],item['arm'],item['scenario_index'])
        if not result['passed']:
            raise ValueError('smoke_prerequisite_not_passed:'+item['experiment'])
    original_idle = launcher.idle_gpus
    original_audit = pool_acceptance.audit_pool
    def idle(reserved):
        if shutil.disk_usage(ROOT).free < 5.5*2**30:
            raise RuntimeError('insufficient_disk_user_action_required')
        return original_idle(reserved)
    launcher.idle_gpus = idle
    pool_acceptance.audit_pool = audit_four_arm
    try:
        launcher.main()
    except Exception as error:
        output = getattr(launcher.main, 'created_output', None)
        if output is not None:
            launcher.save(output/'launcher_result.json', dict(status='launcher_failure', error_type=type(error).__name__, finished_at=time.time()))
        raise
    finally:
        launcher.idle_gpus = original_idle
        pool_acceptance.audit_pool = original_audit


if __name__ == '__main__':main()
