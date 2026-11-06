"""CPU串行执行两个10帧短回放；保留失败，不重试、不新增规划请求。"""
from pathlib import Path
import json,os,subprocess,sys,time

ROOT=Path('/home2/zhaoyx/scenario-dreamer');NAME='anchor_refinement_fixed_plan_window074_20261030_r1'


def main():
    out=ROOT/'experiments'/NAME
    if out.exists():raise FileExistsError('preserve_previous_replay')
    out.mkdir();results=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    for mode in ['legacy_dense_huber','dense_squared_proximal']:
        command=[sys.executable,str(Path(__file__).with_name('window_replay.py')),'--output',str(out/mode),'--mode',mode]
        with (out/f'{mode}.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            (out/f'{mode}_pid.json').write_text(json.dumps(dict(pid=child.pid,command=command,started_at=time.time()),indent=2))
            code=child.wait()
        results.append(dict(mode=mode,exit_code=code))
        (out/'launcher_result.partial.json').write_text(json.dumps(results,indent=2))
        if code:break
    (out/'launcher_result.json').write_text(json.dumps(dict(groups=results,status='completed' if all(x['exit_code']==0 for x in results) else 'failed'),indent=2))


if __name__=='__main__':main()
