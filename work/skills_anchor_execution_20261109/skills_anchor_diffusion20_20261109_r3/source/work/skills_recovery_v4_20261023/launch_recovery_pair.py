"""复用已验证的空闲GPU排队器；仅新目录、新冻结源码，终态不自动重试。"""
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
import launch_pool_pair as launcher
import pool_acceptance
from audit_recovery_v4 import audit_v4


def check_disk():
    # 上轮完整20次约0.13GB；预留0.5GB并保护至少5GB余量，不清理任何历史。
    free=shutil.disk_usage(ROOT).free
    if free<5.5*2**30:raise RuntimeError('insufficient_free_disk_below_5_5GiB; user_action_required')


def main():
    check_disk();original_idle=launcher.idle_gpus;original_audit=pool_acceptance.audit_pool
    def idle(reserved):check_disk();return original_idle(reserved)
    launcher.idle_gpus=idle;pool_acceptance.audit_pool=audit_v4
    try:launcher.main()
    except Exception as error:
        output=getattr(launcher.main,'created_output',None)
        if output is not None:
            launcher.save(output/'launcher_result.json',dict(status='launcher_failure',error_type=type(error).__name__,
                user_action_required='disk' if isinstance(error,RuntimeError) and str(error).startswith('insufficient_free_disk') else None,
                finished_at=time.time()))
        raise
    finally:launcher.idle_gpus=original_idle;pool_acceptance.audit_pool=original_audit


if __name__=='__main__':main()
