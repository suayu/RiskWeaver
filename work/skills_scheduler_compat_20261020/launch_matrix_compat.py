"""兼容服务器Python缺少pidfd_open的构建；只修复等待，不修改实验方法。"""
import ctypes
import errno
import os
import platform
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT))


def pidfd_open_compat(pid,flags=0):
    """Linux x86_64公开系统调用，只获取退出通知句柄，不发送进程信号。"""
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise OSError(errno.ENOSYS,'pidfd compatibility supports Linux x86_64 only')
    libc=ctypes.CDLL(None,use_errno=True)
    syscall=libc.syscall;syscall.restype=ctypes.c_long
    result=syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(flags))
    if result>=0:return int(result)
    error=ctypes.get_errno()
    if error==errno.ESRCH:
        # 前驱恰好退出的竞态：返回已到EOF的可读句柄，让原入口立即重验终态。
        read_fd,write_fd=os.pipe();os.close(write_fd);return read_fd
    raise OSError(error,os.strerror(error))


def main():
    if not hasattr(os,'pidfd_open'):os.pidfd_open=pidfd_open_compat
    from experiments.skills.launch_matrix import main as original_main
    original_main()


if __name__=='__main__':main()
