"""原40帧与对应真实地图上重新构建基础Skills目录，零API、零仿真。"""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import socket
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
OUT=ROOT/'work/skills_rebuilt_current40_20261104_r1'
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_catalog250_20261102')]
from audit import initial_environment
import numpy as np
from omegaconf import OmegaConf
from policies.skills_planner import SkillsAttackPlanner
from policies.llm_adversarial_planner import LLMAdversarialPlanner
from policies.intent_validation import validate_world_intent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prohibited(*args,**kwargs):
    raise AssertionError('API_and_simulation_prohibited')


def main():
    OUT.mkdir(exist_ok=False)
    # 不允许网络或env.step；空CUDA可见集由启动命令额外固定。
    socket.socket.connect=prohibited
    socket.create_connection=prohibited
    inputs=ROOT/'work/skills_stage_a_verification_20261023/original40_inputs.json'
    backup=ROOT/'work/profile_gate_removal_20261104/ego_profile.before.py'
    cfgpath=ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml'
    sourcepaths=[inputs,backup,cfgpath,Path(__file__),ROOT/'policies/skills_planner.py',
                 ROOT/'policies/ego_profile.py',ROOT/'policies/intent_validation.py',
                 ROOT/'policies/llm_adversarial_planner.py',ROOT/'work/skills_catalog250_20261102/audit.py']
    before={str(p):sha(p) for p in sourcepaths}
    spec=importlib.util.spec_from_file_location('old_profile_rebuild40',backup)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    cfg=OmegaConf.load(cfgpath)
    raw=json.loads(inputs.read_text());records=raw['records'];assert len(records)==40
    manifest=json.loads((ROOT/'work/skills_omni_dynamic_pair_protocol_20261104_r1/manifest.json').read_text())
    lookup=dict(zip(manifest['original_scenario_indices'],zip(manifest['scenario_files'],manifest['scenario_sha256'])))
    planner=LLMAdversarialPlanner(provider='dashscope',model_name='offline-audit',model_names=['offline-audit'],client=NS())
    planner.use_obstacles=False;planner._request_attack_plan=prohibited
    geometries={};rows=[];initial_checks=[]
    for item in records:
        scene=int(item['scene'].rsplit('_',1)[1]);filename,digest=lookup[scene]
        file=Path(filename);assert sha(file)==digest
        if scene not in geometries:
            geometry=pickle.loads(file.read_bytes());geometry['_audit_filename']=file.name
            geometries[scene]=geometry
        env=initial_environment(geometries[scene],cfg)
        frame=item['frame'];state=item['state']
        if item['step']==0:
            check=dict(scene=scene,ego_equal=np.array_equal(env.ego_state,np.asarray(frame['ego'])),
                       agents_equal=np.array_equal(env.data_dict['agent'][-1],np.asarray(frame['agents'])),
                       active_equal=np.array_equal(env.agent_active,np.asarray(frame['active'])))
            initial_checks.append(check)
            assert all(check[k] for k in ['ego_equal','agents_equal','active_equal']),check
        # 使用观测帧更新当前运动状态，地图仍是对应SHA核验源的真实地图。
        env.ego_state=np.asarray(frame['ego'],float)
        env.data_dict['agent']=[np.asarray(frame['agents'],float)]
        env.agent_active=np.asarray(frame['active'],bool)
        env.t=item['step'];env.step=prohibited
        env.get_state_for_planning=lambda state=state:state
        row=dict(scene=scene,step=item['step'],active=int(env.agent_active.sum()),
                 visible=len(state['agents']),modes={})
        for expansion in [False,True]:
            cfg.sim.llm.skills.candidate_expansion_enabled=expansion
            cfg.sim.ego_profile.candidate_expansion_enabled=expansion
            for previous in [True,False]:
                engine=SkillsAttackPlanner(cfg.sim,4)
                if previous:engine.builder=old.CandidateBuilder(cfg.sim.ego_profile)
                context=engine.build_context(env,state,planner,profile_context=None)
                checks=[validate_world_intent(planner,state,c) for c in context['candidates']]
                assert all(c['accepted'] for c in checks)
                row['modes'][('expanded' if expansion else 'base')+('_old_gate' if previous else '_new_gate')]=context
        rows.append(row)
        with (OUT/'records.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    assert all(sha(Path(p))==h for p,h in before.items())
    names=['base_old_gate','base_new_gate','expanded_old_gate','expanded_new_gate']
    summary=dict(inputs=len(rows),scenes=len(geometries),initial_checks=initial_checks,
        nonempty={n:sum(bool(r['modes'][n]['candidates']) for r in rows) for n in names},
        source_sha256=before,input_and_source_unchanged=True,API_calls=0,execution_frames=0,D=None,
        scope='Current base Skills build_context on exact historical40 states and original real maps; not full temporal factory, LLM acceptance, or execution success',
        provenance='current source vs pre-gate-removal CandidateBuilder; same map and shared current numeric validation',
        old_historical_catalogs_not_reused=True)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=lambda x:bool(x)),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_sha256','initial_checks']},ensure_ascii=False))


if __name__=='__main__':
    main()
