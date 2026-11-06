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

for i in (59,244):
    name=m['scenario_files'][i]
    g=pickle.loads(Path(name).read_bytes())
    g['_audit_filename']=Path(name).name
    env=initial_environment(g,cfg)
    cfg.sim.llm.skills.candidate_expansion_enabled=True
    engine=SkillsAttackPlanner(cfg.sim)
    ctx=engine.build_context(env,env.get_state_for_planning(),low)
    from collections import Counter
    print(i, 'screen',ctx.get('candidate_rejections'), 'semantic',dict(Counter(x.get('reason') for x in ctx['semantic_precheck'])))
