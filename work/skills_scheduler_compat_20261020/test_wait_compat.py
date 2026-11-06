"""使用真实内核退出事件验证兼容入口；不读取或改写任何仿真实验。"""
import os
import select
import subprocess
import sys
import pytest
from launch_matrix_compat import pidfd_open_compat


pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Linux kernel pidfd test')


def test_actual_child_exit_makes_handle_readable():
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(0.15)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    descriptor=pidfd_open_compat(process.pid)
    try:
        ready,_,_=select.select([descriptor],[],[],3.)
        assert ready==[descriptor]
        assert process.wait(timeout=3)==0
    finally:os.close(descriptor)


def test_predecessor_already_gone_does_not_wait_ninety_minutes():
    descriptor=pidfd_open_compat(2147483647)
    try:assert select.select([descriptor],[],[],.1)[0]==[descriptor]
    finally:os.close(descriptor)


def test_invalid_flags_are_not_silently_treated_as_completion():
    with pytest.raises(OSError):pidfd_open_compat(os.getpid(),0xffffffff)
