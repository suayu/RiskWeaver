# 完整真实入口发送前拦截：核验无引导、无检索、自然终止与有限缓冲区，不调用付费API。
from pathlib import Path
import json,os,subprocess,sys
ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=ROOT/'work/skills_natural_rollback_20261121'
sys.path[:0]=[str(ROOT),str(HERE)]
from campaign import environment,load,dump,verify_sources
code=r'''
from pathlib import Path
import json,sys,os,runpy,httpx
from omegaconf import OmegaConf
root=Path('/home2/zhaoyx/scenario-dreamer');here=root/'work/skills_natural_rollback_20261121'
sys.path[:0]=[str(root),str(here)]
from campaign import structured_keys
command=json.loads((here/'protocol/protocol.json').read_text())['template']
command[command.index('--manifest')+1]=str(here/'protocol/smoke_manifest.json')
command[command.index('--output-dir')+1]=str(here/'preflight_v5/skills')
os.environ['RW_NL_PREFERENCE_ENABLED']='1';os.environ['RW_NL_INSTRUCTION']='生成追尾的危险场景'
class ProbeDone(BaseException):pass
receipt={}
def send(client,request,*a,**kw):
    payload=json.loads(request.content);text=json.dumps(payload,ensure_ascii=False)
    cfg=OmegaConf.load(here/'preflight_v5/skills/resolved_config.yaml')
    natural=json.loads((here/'preflight_v5/skills/natural_termination_protocol.json').read_text())
    checks=dict(model=payload['model']=='deepseek-flash',
        no_nl=('natural_language_preference' not in text and 'preference_assessment' not in text and 'rw-nl-preference/SKILL.md' not in text),
        no_rag=('retrieved_attack_evidence' not in structured_keys(payload['messages'])[0] and structured_keys(payload['messages'])[1]>0 and not cfg.sim.rag.enabled),
        observation=cfg.sim.llm_road_observation_radius_m==60 and cfg.sim.planning_observation_radius_m==45,commonroad=cfg.sim.evaluation.reachability.backend=='commonroad',joint=cfg.sim.llm.attack_mode=='joint',static_fallback=bool(cfg.sim.llm.skills.static_fallback_enabled),
        no_profile=not cfg.sim.traffic_model.iterative_adversarial.profile_enabled,
        natural=(natural['effective_frame_cap'] is None and natural['allocation_horizon']==400 and natural['runner_frame_cap_removed'] and natural['simulator_frame_cap_removed']))
    receipt.update(verified=all(checks.values()),checks=checks,network_calls=0,env_steps=0,model=payload['model'],natural_termination=natural)
    raise ProbeDone()
httpx.Client.send=send
sys.argv=command[1:]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except ProbeDone:pass
assert receipt,'no real request intercepted'
assert receipt['verified'],str(receipt)
entry=json.loads((here/'preflight_v5/skills/generation_mode.json').read_text())
assert entry['extra_language_guidance'] is False
(here/'preflight_verified.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''
verify_sources(load(HERE/'protocol/source_sha256.json'))
(HERE/'preflight_v5').mkdir(exist_ok=False)
dump(HERE/'preflight_v5/source_sha256.json',load(HERE/'protocol/source_sha256.json'))
with (HERE/'preflight_v5.log').open('x')as log:
    p=subprocess.run(['bash','scripts/run_with_llm_provider.sh','deepseek',sys.executable,'-'],input=code.encode(),cwd=ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,timeout=240)
assert p.returncode==0,'see preflight.log'
print(json.dumps(load(HERE/'preflight_verified.json')))
