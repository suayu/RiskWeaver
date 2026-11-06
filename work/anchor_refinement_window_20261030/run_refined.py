"""可回退的独立运行入口；显式选择后才装配，不修改公共源码或已有实验。"""
from pathlib import Path
import argparse,runpy,sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement',choices=['off','squared','proximal'],default='off')
    parser.add_argument('script',type=Path)
    parser.add_argument('script_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    path=args.script.resolve()
    if not path.is_file():raise FileNotFoundError(path)
    # 原脚本仍负责资源准入、预算、输入冻结和结果保存；此入口不创建新队列。
    previous_argv,previous_path=sys.argv,list(sys.path)
    try:
        if args.refinement!='off':from objective import install_anchor_objective
        sys.argv=[str(path),*args.script_args];sys.path.insert(0,str(path.parent))
        if args.refinement=='off':
            runpy.run_path(str(path),run_name='__main__');return
        mode='dense_squared_proximal' if args.refinement=='proximal' else 'dense_squared'
        with install_anchor_objective(mode):runpy.run_path(str(path),run_name='__main__')
    finally:sys.argv=previous_argv;sys.path[:]=previous_path


if __name__=='__main__':main()
