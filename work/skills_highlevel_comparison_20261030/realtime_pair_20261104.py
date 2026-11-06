import hashlib,json,os,shutil
from urllib.parse import urlsplit
from omegaconf import OmegaConf
import frozen_planning as base
from pathlib import Path
from work.realtime_planner_20261104.transport import RealtimeClient
root=base.ROOT
prior=root/'work/skills_highlevel_legacy_skills_protocol_20261030_r3'
out=root/'experiments/skills_realtime_fixed_highlevel_pair_20261104_r1'
model='qwen3.8-omni-flash-realtime'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
samples=json.loads((prior/'inputs.json').read_text())
sample=next(s for s in samples if s['scene']==74)
cfg=OmegaConf.load(prior/'resolved_config.yaml')
out.mkdir()
shutil.copyfile(prior/'resolved_config.yaml',out/'resolved_config.yaml')
base.save(out/'input.json',sample)
names=set(json.loads((prior/'source_sha256.json').read_text()))
names.add(Path(__file__).relative_to(root).as_posix())
names.update(["work/realtime_planner_20261104/transport.py","work/realtime_planner_20261104/test_transport.py"])
hashes={}
for name in sorted(names):
 data=(root/name).read_bytes()
 hashes[name]=hashlib.sha256(data).hexdigest()
 target=out/'source_snapshot'/name
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(data)
base.save(out/'source_sha256.json',hashes)
base.save(out/'protocol.json',dict(model=model,provider='dashscope',scene=74,planned=2,free_protection_user_confirmed=True,max_physical_per_arm=4,scope='frozen_dynamic_initial_catalog_replay'))
def factory(**kw):
 planner=base.LLMAdversarialPlanner(**kw)
 assert planner.client and urlsplit(str(planner.client.base_url)).hostname=='dashscope.aliyuncs.com'
 planner.client=RealtimeClient(os.environ['DASHSCOPE_API_KEY'],out/'transport_calls',timeout=180)
 return planner
rows=[]
restore=base.install_control('service_only')
try:
 for mode in ('legacy_off','temporal_skills_common'):
  assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
  row=base.run_trial(sample,mode,[model],cfg,out/mode,planner_factory=factory)
  rows.append({k:row.get(k) for k in ['mode','accepted_plan','planner_service_failure','runtime_failure','validation_reason','logical_requests']})
  base.save(out/'partial.json',rows)
  if row.get('planner_service_failure') or row.get('runtime_failure'):break
finally:
 restore()
 base.save(out/'summary.json',dict(planned=2,attempted=len(rows),rows=rows,D=None,execution_frames=0))
print(json.dumps(rows,ensure_ascii=False))
