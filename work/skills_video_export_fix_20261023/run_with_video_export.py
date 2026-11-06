"""下一轮可选整数视频入口；默认旧出口，不修改当前冻结运行器。"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
import run_pool_validation as base
from prospective_video_hook import install_integer_video_export


def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--video-export',choices=('legacy','integer'),default='legacy')
    args,remaining=parser.parse_known_args()
    output=Path(remaining[remaining.index('--output-dir')+1])
    if output.exists():raise FileExistsError('preserve_historical_output')
    files=[Path(__file__).resolve(),Path(__file__).with_name('prospective_video_hook.py'),Path(__file__).with_name('integer_frame_export.py'),ROOT/'utils/viz.py']
    protocol=dict(video_export=args.video_export,fps=5,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='future fresh run only, common exporter for both arms; no algorithm/safety/metric changes')
    sys.argv=[sys.argv[0]]+remaining
    if '--prepare-only' in remaining:
        # 预检保持一个JSON对象且零API，不创建结果目录或安装运行钩子。
        text=io.StringIO()
        with redirect_stdout(text):base.main()
        result=json.loads(text.getvalue());result['video_export_protocol']=protocol
        print(json.dumps(result,ensure_ascii=False));return
    # 新入口必须由包含出口源码的新冻结协议启动，拒绝套用当前R2旧快照。
    source_manifest=output.parent/'source_sha256.json'
    if not source_manifest.is_file():raise ValueError('new_frozen_source_manifest_required')
    frozen=json.loads(source_manifest.read_text())
    if any(frozen.get(name)!=digest for name,digest in protocol['source_sha256'].items()):
        raise ValueError('video_export_sources_not_frozen_or_changed')
    restore=install_integer_video_export(output) if args.video_export=='integer' else lambda:None
    try:base.main()
    finally:
        restore()
        if output.exists():
            with (output/'video_export_protocol.json').open('x') as stream:json.dump(protocol,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':main()
