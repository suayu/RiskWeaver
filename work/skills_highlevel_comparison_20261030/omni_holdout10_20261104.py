import hashlib,json,os,shutil
from urllib.parse import urlsplit
from omegaconf import OmegaConf
import frozen_planning as base
from pathlib import Path
root=base.ROOT
prior=root/'work/skills_candidate_local_holdout_protocol_20261030_r1'
out=root/'experiments/skills_omni_fixed_highlevel_holdout10_20261104_r1'
model='qwen3.8-omni-flash'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
samples=json.loads((prior/'inputs.json').read_text())
assert [s['scene'] for s in samples]==[6,28,45,102,114,143,146,185,229,232]
cfg=OmegaConf.load(prior/'resolved_config.yaml')
out.mkdir()
shutil.copyfile(prior/'resolved_config.yaml',out/'resolved_config.yaml')
base.save(out/'inputs.json',samples)
names=set(json.loads((prior/'source_sha256.json').read_text()))
names.add(Path(__file__).relative_to(root).as_posix())
hashes={}
for name in sorted(names):
 data=(root/name).read_bytes()
 hashes[name]=hashlib.sha256(data).hexdigest()
 target=out/'source_snapshot'/name
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(data)
base.save(out/'source_sha256.json',hashes)
base.save(out/'protocol.json',dict(model=model,provider='dashscope',scenes=[s["scene"] for s in samples],planned=20,free_protection_user_confirmed=True,max_physical_per_arm=4,scope='frozen_dynamic_initial_catalog_replay'))
def factory(**kw):
 planner=base.LLMAdversarialPlanner(**kw)
 assert planner.client and urlsplit(str(planner.client.base_url)).hostname=='dashscope.aliyuncs.com'
 planner.client=planner.client.with_options(timeout=90,max_retries=0)
 return planner
rows=[]
restore=base.install_control('service_only')
try:
 for index,sample in enumerate(samples):
  modes=('legacy_off','temporal_skills_common')
  if index%2: modes=modes[::-1]
  for mode in modes:
   assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
   row=base.run_trial(sample,mode,[model],cfg,out/f'{sample["scene"]:03d}_{mode}',planner_factory=factory)
   rows.append({k:row.get(k) for k in ['scene','mode','accepted_plan','planner_service_failure','runtime_failure','validation_reason','logical_requests']})
   base.save(out/'partial.json',rows)
   if row.get('planner_service_failure') or row.get('runtime_failure'): raise RuntimeError('terminal_trial_failure')
finally:
 restore()
 base.save(out/'summary.json',dict(planned=20,attempted=len(rows),rows=rows,D=None,execution_frames=0))
print(json.dumps(rows,ensure_ascii=False))
