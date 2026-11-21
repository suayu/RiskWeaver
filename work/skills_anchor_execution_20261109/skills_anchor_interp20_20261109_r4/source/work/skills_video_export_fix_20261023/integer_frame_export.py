"""按整数帧序列生成派生视频；不修改实验原件、不依赖浮点累计时长。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import numpy as np
from PIL import Image
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_sources(source_dir,trace):
    """图像编号必须与完整trace状态一一对应；不忽略缺帧、重复或额外PNG。"""
    images=[]
    for p in source_dir.glob('*.png'):
        match=re.fullmatch(r'frame_(\d+)\.png',p.name)
        if match is None:raise ValueError('unexpected_png_name')
        images.append((int(match.group(1)),p))
    images.sort(key=lambda x:x[0])
    indices=[x[0] for x in images]
    if not images or indices!=list(range(len(images))):raise ValueError('noncontiguous_or_duplicate_png_indices')
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    states=[r['step'] for r in rows if r.get('kind')=='state']
    if states!=indices:raise ValueError('trace_and_png_sequence_mismatch')
    return [x[1] for x in images]


def encode(source_dir,trace,output,experiment_root,fps=5):
    source_dir=Path(source_dir).resolve();trace=Path(trace).resolve();output=Path(output).resolve()
    experiment_root=Path(experiment_root).resolve()
    if not source_dir.is_relative_to(experiment_root) or not trace.is_relative_to(experiment_root):raise ValueError('source_outside_experiment')
    if output.is_relative_to(experiment_root):raise ValueError('derived_output_must_be_outside_experiment')
    if not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps<=0:raise ValueError('invalid_fps')
    partial=output.with_name(output.stem+'.partial.mp4');receipt=output.with_suffix('.manifest.json')
    if any(p.exists() for p in [output,partial,receipt]):raise FileExistsError('preserve_existing_output')
    images=frame_sources(source_dir,trace)
    before={str(p):digest(p) for p in [trace,*images]}
    with Image.open(images[0]) as first:size=first.size
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=FFMPEG_VideoWriter(str(partial),size,fps,codec='libx264',threads=1)
    try:
        # 直接写N次，完全绕过MoviePy的duration与arange时间采样。
        for p in images:
            with Image.open(p) as img:
                if img.size!=size:raise ValueError('image_size_changed')
                writer.write_frame(np.asarray(img.convert('RGB')))
    finally:writer.close()
    after={str(p):digest(p) for p in [trace,*images]}
    if before!=after:raise ValueError('source_changed_during_export')
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=nb_read_frames,width,height','-of','json',str(partial)],capture_output=True,text=True,check=True)
    video=json.loads(probe.stdout)['streams'][0]
    if int(video['nb_read_frames'])!=len(images):raise ValueError('encoded_frame_count_mismatch')
    if output.exists():raise FileExistsError('output_appeared_during_export')
    partial.rename(output)
    result=dict(status='derived_export_verified',source_directory=str(source_dir),trace=str(trace),
        source_sha256=before,source_stable=True,output=str(output),output_sha256=digest(output),fps=fps,
        source_png_count=len(images),trace_state_count=len(images),decoded_frame_count=int(video['nb_read_frames']),
        one_write_per_source_image=True,source_step_to_video_index=[dict(source_step=i,video_index=i,image=p.name) for i,p in enumerate(images)],
        original_experiment_unchanged=True,original_protocol_gate_overridden=False,
        scope='derived visualization only; does not amend original acceptance, simulation, plans, D or performance')
    with receipt.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--fps',type=float,default=5)
    a=p.parse_args();r=encode(a.source_dir,a.trace,a.output,a.experiment_root,a.fps)
    print(json.dumps({k:v for k,v in r.items() if k not in ['source_sha256','source_step_to_video_index']},ensure_ascii=False))


if __name__=='__main__':main()
