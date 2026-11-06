"""七个既定真实初态的旧自由/Skills共同门控接线；所有回复为显式固定替身。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import pickle
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from omegaconf import OmegaConf
from policies.hybrid_skills_planner import StaticCombinationPlanner
from policies.ego_profile import clean_json
from work.skills_common_static_gate_20261102.common_gate import common_scope,plan_once,state_hash
from work.skills_common_static_gate_20261102.test_common_gate import legacy_planner
from work.skills_static_physical15_20261102.audit_seven_wiring import (
    fixture_planner,initial_environment,verify_sources)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    content=json.dumps(clean_json(value),ensure_ascii=False,allow_nan=False,indent=2)
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x',encoding='utf-8') as file:file.write(content)
    pending.replace(path)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only')
    out=HERE/'seven_common_r1'
    if out.exists():raise FileExistsError('preserve_existing_common_audit')
    source=ROOT/'work/skills_static_physical15_20261102/seven_wiring_r2'
    if json.loads((source/'stage_result.json').read_text())['status']!='completed':
        raise ValueError('previous_wiring_evidence_not_complete')
    scenes=[91,100,129,211,238,241,246]
    manifest_path=source/'frozen250_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    cfg=OmegaConf.load(source/'resolved_config.yaml')
    protected_path=ROOT/'work/skills_hybrid_closed_loop2_deepflash_protocol_20261102_r4'
    protected=json.loads((protected_path/'source_sha256.json').read_text())
    protocol=json.loads((protected_path/'protocol.json').read_text())
    if not verify_sources(protocol,protected)['verified']:raise ValueError('protected_source_changed')
    dependencies=[p for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')]
    dependencies.extend(ROOT/f'work/skills_static_physical15_20261102/{name}'
        for name in ['adapter.py','geometry.py','audit_seven_wiring.py','planner.py','manifest.json','SKILL.md'])
    dependencies += [ROOT/'work/hybrid_public_integration_20261101/test_public_hybrid.py',
                     ROOT/'work/skills_catalog250_20261102/audit.py']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies}
    out.mkdir()
    for name in hashes:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    save(out/'protocol.json',dict(scenes=scenes,planned_scenes=7,planned_path_requests=14,
        source_sha256=hashes,original250_manifest_sha256=sha(manifest_path),
        config_sha256=sha(source/'resolved_config.yaml'),fixture=True,
        fixed_rule_common_to_both_paths=True,physical_API_requests=0,env_steps=0,D=None,
        scope='static_highlevel_wiring_real_maps_not_full_temporal_factory_or_dynamic_routing',
        not_an_independent_generalization_sample=True,goal_complete=False))
    (out/'frozen250_manifest.json').write_bytes(manifest_path.read_bytes())
    (out/'resolved_config.yaml').write_bytes((source/'resolved_config.yaml').read_bytes())
    save(out/'protected_source_sha256.json',protected)
    results=[];attempted=0
    try:
        for index in scenes:
            if any(sha(ROOT/name)!=digest for name,digest in hashes.items()):raise ValueError('audit_source_changed')
            filename=manifest['scenario_files'][index]
            if sha(filename)!=manifest['scenario_sha256'][index]:raise ValueError('scenario_changed')
            attempted+=1
            geometry=pickle.loads(Path(filename).read_bytes());geometry['_audit_filename']=Path(filename).name
            env=initial_environment(geometry,cfg)
            def forbidden(*args,**kwargs):raise AssertionError('env_step_forbidden')
            env.step=forbidden
            state=copy.deepcopy(env.get_state_for_planning());before=state_hash(state)
            low,skills_calls=fixture_planner()
            engine=StaticCombinationPlanner(cfg.sim,4)
            with common_scope(low,env,True):
                candidates=engine.build_context(env,state,low)['obstacle_candidates']
            if not candidates:raise ValueError('fixed_recovered_case_has_no_catalog')
            requested=candidates[0]['placements']
            old,old_calls=legacy_planner(env,requested)
            legacy=plan_once(old,env,state,['无可用车辆时静态组合拦路'],builder=engine.builder,enabled=True)
            skills=plan_once(low,env,state,['无可用车辆时静态组合拦路'],skills_engine=engine,builder=engine.builder,enabled=True)
            a=legacy['final_common_static_validation'];b=skills['final_common_static_validation']
            assert a and b and a['accepted'] and b['accepted']
            assert len(old_calls)==len(skills_calls)==1
            assert legacy['common_rule_sha256']==skills['common_rule_sha256']
            assert '实验共同静态校验规则' in old_calls[0]
            # 结构化模型请求中也必须真实携带相同规则，而不是仅记录计划装配意图。
            assert '实验共同静态校验规则' in json.dumps(skills_calls,ensure_ascii=False)
            assert before==state_hash(env.get_state_for_planning())
            row=dict(scene=index,scenario_sha256=manifest['scenario_sha256'][index],
                source_state_sha256=before,catalog_count=len(candidates),
                legacy=dict(evaluation=legacy,mocked_requests=old_calls,trace=old.last_trace),
                skills=dict(evaluation=skills,mocked_requests=skills_calls,trace=low.last_trace),
                legacy_fixture_not_free_model_reasoning=True,skills_fixture_not_LLM=True,
                physical_API_requests=0,env_steps=0,D=None,collision=None)
            save(out/f'scene{index:03d}.json',row);results.append(row)
        integrity=verify_sources(protocol,protected)
        save(out/'runtime_source_integrity.json',integrity)
        assert integrity['verified']
        save(out/'stage_result.json',dict(status='completed',planned_scenes=7,attempted_scenes=attempted,
            completed_scenes=len(results),path_requests=14,mock_legacy_static_valid=7,mock_skills_static_valid=7,
            physical_API_requests=0,env_steps=0,fresh_LLM_coverage=None,D=None,collision=None,goal_complete=False))
    except Exception as error:
        save(out/'stage_result.json',dict(status='failed',planned_scenes=7,attempted_scenes=attempted,
            completed_scenes=len(results),unattempted_scenes=7-attempted,error_type=type(error).__name__,
            error=str(error),physical_API_requests=0,env_steps=0,D=None,collision=None,goal_complete=False))
        raise


if __name__=='__main__':main()
