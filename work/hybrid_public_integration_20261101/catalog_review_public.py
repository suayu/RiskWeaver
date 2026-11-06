"""原开发/复核二十初态的目录覆盖审计；不调用API、不执行仿真，不改变历史动态目录。"""
from pathlib import Path
from types import SimpleNamespace as NS
import copy,hashlib,json,pickle,sys
import numpy as np
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'work/skills_highlevel_comparison_20261030')]
import frozen_planning as base
from policies.ego_profile import CandidateBuilder
from policies.static_attack_catalog import build_static_combinations


def main():
    out=HERE/'catalog_review20_r1'
    out.mkdir(exist_ok=False)
    cfg=OmegaConf.load(ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml')
    cfg.sim.llm.skills.grounding_validation_mode='candidate_local'
    planner=base.LLMAdversarialPlanner(model_name='glm-5.3',model_names=['glm-5.3'],provider='dashscope',client=NS())
    planner.use_obstacles=True;builder=CandidateBuilder(cfg.sim.ego_profile)
    rows=[]
    for stage,proto,manifest_path in [
        ('development','work/skills_candidate_local_development_protocol_20261030_r4','experiments/skills_temporal_development10_paired_20261029_r1/manifest.json'),
        ('previously_examined_review','work/skills_candidate_local_holdout_protocol_20261030_r1','work/skills_candidate_local_holdout_protocol_20261030_r1/validation10.json')]:
        path=ROOT/proto/'inputs.json';samples=json.loads(path.read_text())
        manifest=json.loads((ROOT/manifest_path).read_text())
        if len(samples)!=10 or [s['scene'] for s in samples]!=manifest['original_scenario_indices']:
            raise ValueError('original_frozen_ten_changed')
        for sample in samples:
            index=manifest['original_scenario_indices'].index(sample['scene'])
            scene_path=Path(manifest['scenario_files'][index]);raw=scene_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=manifest['scenario_sha256'][index]:raise ValueError('scene_pickle_changed')
            geometry=pickle.loads(raw);frame=sample['frame'];state=copy.deepcopy(sample['state'])
            env=NS(current_step=sample['source_step'],current_scene_id=sample['scene_id'],dt=float(cfg.sim.dt),
                ego_state=np.asarray(frame['ego']),data_dict={'agent':[np.asarray(frame['agents'])]},
                agent_active=np.asarray(frame['active']),scenario_dict=geometry,
                get_static_obstacles=lambda:state.get('static_obstacles',[]),get_state_for_planning=lambda:state)
            dynamic=sample['contexts']['temporal_original']['candidates']
            catalog,audit=([],[]) if dynamic else build_static_combinations(env,planner,builder)
            row=dict(stage=stage,scene=sample['scene'],source_step=sample['source_step'],visible_vehicles=len(state['agents']),
                original_dynamic_candidates=len(dynamic),static_combinations=len(catalog),
                branch='dynamic' if dynamic else 'static' if catalog else 'unavailable',
                candidate_coverage=bool(dynamic or catalog),source_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scenario_sha256=hashlib.sha256(raw).hexdigest(),static_catalog=catalog,static_audit=audit,
                accepted_LLM_plan=False,scope='catalog_availability_only_not_actual_LLM_acceptance_or_threat')
            rows.append(row)
            (out/f'{stage}_scene{sample["scene"]:03d}.json').write_text(json.dumps(row,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    summary=dict(planned=20,attempted=len(rows),covered=sum(r['candidate_coverage'] for r in rows),
        dynamic=sum(r['branch']=='dynamic' for r in rows),static=sum(r['branch']=='static' for r in rows),
        unavailable=[dict(stage=r['stage'],scene=r['scene']) for r in rows if not r['candidate_coverage']],
        API_requests=0,new_execution_frames=0,D=None,goal_complete=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':main()
