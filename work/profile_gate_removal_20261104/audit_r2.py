import sys, json, pickle, hashlib, importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
root=Path.cwd()
sys.path[:0]=[str(root),str(root/'work/skills_catalog250_20261102')]
from audit import initial_environment
from omegaconf import OmegaConf
from policies.skills_planner import SkillsAttackPlanner
from policies.llm_adversarial_planner import LLMAdversarialPlanner
here=Path(__file__).parent
spec=importlib.util.spec_from_loader('old_profile',loader=None)
old=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=old
exec(compile((here/'ego_profile.before.py').read_bytes(),'ego_profile.before.py','exec'),old.__dict__)
source=root/'work/skills_catalog250_20261102/all250_r1/frozen_manifest.json'
m=json.loads(source.read_text())
cfg=OmegaConf.load(root/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
low=LLMAdversarialPlanner(model_name='offline-audit',model_names=['offline-audit'],provider='dashscope',client=NS())
low.use_obstacles=False
def forbidden(*a,**k): raise AssertionError('no API or simulation')
low._request_attack_plan=forbidden
rows=[]
with (here/'coverage_r2.jsonl').open('x') as f:
 for i,(name,digest) in enumerate(zip(m['scenario_files'],m['scenario_sha256'])):
    raw=Path(name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==digest
    g=pickle.loads(raw)
    g['_audit_filename']=Path(name).name
    env=initial_environment(g,cfg)
    env.step=forbidden
    state=env.get_state_for_planning()
    result=[]
    for expansion in [False,True]:
        cfg.sim.llm.skills.candidate_expansion_enabled=expansion
        cfg.sim.ego_profile.candidate_expansion_enabled=expansion
        for previous in [True,False]:
            engine=SkillsAttackPlanner(cfg.sim)
            if previous: engine.builder=old.CandidateBuilder(cfg.sim.ego_profile)
            context=engine.build_context(env,state,low)
            result.append(len(context['candidates']))
    row=dict(scene=i,active=int(env.agent_active.sum()),visible=len(state['agents']),counts=result)
    rows.append(row)
    f.write(json.dumps(row)+'\n')
    f.flush()
assert len(rows)==250
summary=dict(scenes=len(rows),columns=['base_old','base_new','expanded_old','expanded_new'],coverage=[sum(r['counts'][j]>0 for r in rows) for j in range(4)],empty_new=[r['scene'] for r in rows if not r['counts'][3]],API_requests=0)
(here/'summary_r2.json').write_text(json.dumps(summary,indent=2))
print(summary)
