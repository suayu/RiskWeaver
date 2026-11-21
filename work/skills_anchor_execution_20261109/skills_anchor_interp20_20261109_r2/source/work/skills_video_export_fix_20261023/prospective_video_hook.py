"""未来新实验专用的可撤销视频出口；禁止挂到已存在的结果目录。"""
import hashlib
import json
from pathlib import Path
import re
import shutil
from integer_frame_export import encode


def install_integer_video_export(run_root,runner=None):
    run_root=Path(run_root).resolve()
    stage=run_root.with_name(run_root.name+'_integer_video_exports')
    if run_root.exists() or stage.exists():raise FileExistsError('new_run_directory_required')
    if runner is None:
        from experiments.comparison import run_e1_group as runner
    original=runner.generate_video

    def generate(name,output_dir,delete_images=True):
        source=Path(output_dir).resolve()
        if not source.is_relative_to(run_root) or not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('video_path_outside_new_run')
        if delete_images:raise ValueError('preserve_source_images_for_evaluation')
        target=source/f'{name}.mp4';mapping=target.with_suffix('.frames.json')
        if target.exists() or mapping.exists():raise FileExistsError('preserve_existing_video')
        result=encode(source,source/'execution_trace.jsonl',stage/f'{name}.mp4',run_root,fps=5)
        # 复制已核验的新视频，排他创建；不覆盖原件，所有源PNG保留。
        with (stage/f'{name}.mp4').open('rb') as inp,target.open('xb') as dest:shutil.copyfileobj(inp,dest)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=result['output_sha256']:raise ValueError('video_copy_hash_mismatch')
        with mapping.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
        return str(target)

    runner.generate_video=generate
    def restore():runner.generate_video=original
    return restore
