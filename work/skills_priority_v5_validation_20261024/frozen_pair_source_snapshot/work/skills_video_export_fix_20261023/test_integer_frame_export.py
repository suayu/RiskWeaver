"""验证真实编码帧数、278帧浮点边界、缺帧与历史保护，不调用GPU或LLM。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
import pytest
from integer_frame_export import encode


def fixture(tmp_path,n):
    root=tmp_path/'experiment';src=root/'scene';src.mkdir(parents=True)
    for i in range(n):
        image=np.zeros((16,24,3),dtype=np.uint8);image[:,:,0]=i%256;image[:,:,1]=i//256
        Image.fromarray(image).save(src/f'frame_{i:03d}.png')
    trace=src/'execution_trace.jsonl';trace.write_text(''.join(json.dumps(dict(kind='state',step=i))+'\n' for i in range(n)))
    return root,src,trace


def test_actual_moviepy_278_frame_float_boundary_and_integer_fix(tmp_path):
    root,src,trace=fixture(tmp_path,278)
    clip=ImageSequenceClip([str(p) for p in sorted(src.glob('*.png'))],fps=5)
    try:
        assert clip.duration>278/5
        assert sum(1 for _ in clip.iter_frames(fps=5,logger=None))==279
    finally:clip.close()
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
    result=encode(src,trace,tmp_path/'derived'/'scene.mp4',root)
    assert result['decoded_frame_count']==result['source_png_count']==result['trace_state_count']==278
    assert result['source_step_to_video_index'][-1]==dict(source_step=277,video_index=277,image='frame_277.png')
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir()}
    assert not result['original_protocol_gate_overridden']


@pytest.mark.parametrize('n',[1,139])
def test_integer_encoder_handles_short_and_odd_frame_counts(tmp_path,n):
    root,src,trace=fixture(tmp_path,n)
    assert encode(src,trace,tmp_path/'derived.mp4',root)['decoded_frame_count']==n


def test_incomplete_source_is_not_silently_encoded(tmp_path):
    root,src,trace=fixture(tmp_path,3)
    trace.write_text(json.dumps(dict(kind='state',step=0))+'\n')
    with pytest.raises(ValueError,match='trace_and_png_sequence_mismatch'):encode(src,trace,tmp_path/'derived.mp4',root)
    assert not (tmp_path/'derived.mp4').exists()


def test_original_experiment_and_existing_derived_are_protected(tmp_path):
    root,src,trace=fixture(tmp_path,1)
    with pytest.raises(ValueError,match='outside_experiment'):encode(src,trace,src/'new.mp4',root)
    existing=tmp_path/'existing.mp4';existing.write_bytes(b'historical-video')
    with pytest.raises(FileExistsError):encode(src,trace,existing,root)
    assert existing.read_bytes()==b'historical-video'
