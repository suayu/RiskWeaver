"""仅在原围栏首次实际加载检查处等待显存；不重启进程或改采样。"""
from contextlib import contextmanager
from pathlib import Path
import json
import shutil
import sys
import time


@contextmanager
def wait_before_loading(cuda, minimum_free_bytes, report_path, expected_caller, disk_root,
                        wait=time.sleep, interval_seconds=15):
    original = cuda.mem_get_info
    expected_caller = Path(expected_caller).resolve()
    report_path = Path(report_path)
    initial = True

    def record(status, readings, error=None):
        # 收据位于已有实验父目录，不提前创建公共runner拥有的arm目录。
        value = dict(version='actual_worker_admission_wait_v1', status=status,
                     minimum_free_bytes=minimum_free_bytes, last_free_bytes=readings[-1] if readings else None,
                     samples=len(readings), checked_at=time.time(), error_type=error,
                     no_model_or_API_retry=True, original_cap_and_reserve_unchanged=True)
        pending = report_path.with_suffix('.pending')
        pending.write_text(json.dumps(value, indent=2), encoding='utf-8')
        pending.replace(report_path)

    def measured(*args, **kwargs):
        nonlocal initial
        frame = sys._getframe(1)
        if (not initial or frame.f_code.co_name != 'main'
                or Path(frame.f_code.co_filename).resolve() != expected_caller):
            return original(*args, **kwargs)
        readings = []
        try:
            while True:
                actual = original(*args, **kwargs)
                readings.append(int(actual[0]))
                if actual[0] >= minimum_free_bytes:
                    initial = False
                    # 后续资源观测线程和全部模型代码立即恢复原始Torch接口。
                    cuda.mem_get_info = original
                    record('admitted_before_model_load', readings)
                    return actual
                record('waiting_before_model_load', readings)
                if shutil.disk_usage(disk_root).free < 5.5 * 2**30:
                    raise RuntimeError('insufficient_disk_while_waiting_for_memory')
                wait(interval_seconds)
        except BaseException as error:
            record('failed_before_model_load', readings, type(error).__name__)
            raise

    cuda.mem_get_info = measured
    try:
        yield
    finally:
        cuda.mem_get_info = original
