import os,subprocess,json,time,signal
from pathlib import Path
root=Path('/home2/zhaoyx/scenario-dreamer')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH='.:safe-sim:safe-sim/trajdata/src',PYTHONUNBUFFERED='1',LLM_MODEL_NAME='qwen3.8-flash',LLM_MODEL_NAMES='qwen3.8-flash')
with (root/'work/skills_anchor_projection_cpu_probe_20261020.log').open('xb') as log:
 p=subprocess.Popen(['bash','scripts/run_with_llm_provider.sh','dashscope','/home2/zhaoyx/miniconda3/envs/scenario-dreamer/bin/python','work/skills_anchor_projection_dev_20261020/probe_anchor_projection_cpu.py'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (root/'work/skills_anchor_projection_cpu_probe_child_20261020.json').write_text(json.dumps({'pid':p.pid,'started_at':time.time()}))
 try:code=p.wait(timeout=600);timed_out=False
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
(root/'work/skills_anchor_projection_cpu_probe_supervisor_result_20261020.json').write_text(json.dumps({'exit_code':code,'timed_out':timed_out,'finished_at':time.time()}))
