"""原250初态的路线后备开/关目录比较；不调用LLM、不生成轨迹。"""
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'work/skills_catalog250_20261102'),
    str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
from audit import initial_environment,save,sha
from adapter import install
from launch_pool_pair import verify_sources


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_only_required')
    from omegaconf import OmegaConf
    from policies.ego_profile import CandidateBuilder
    from policies.static_attack_catalog import build_static_combinations
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    old=ROOT/'work/skills_catalog250_20261102'
    stage=json.loads((old/'all250_r1/stage_result.json').read_text())
    previous=json.loads((old/'union250_r1/stage_result.json').read_text())
    assert stage['status']==previous['status']=='completed' and stage['scanned']==250
    manifest=json.loads((old/'all250_r1/frozen_manifest.json').read_text())
    live=ROOT/'work/skills_hybrid_closed_loop2_protocol_20261102_r1'
    protocol=json.loads((live/'protocol.json').read_text());protected=json.loads((live/'source_sha256.json').read_text())
    hashes={name:sha(HERE/name) for name in ['geometry.py','adapter.py','scan.py']}
    cfg_path=ROOT/'experiments/skills_rag_two_arm_development2_20261030_r2/skills_no_rag/resolved_config.yaml'
    cfg=OmegaConf.load(cfg_path)
    out=HERE/'all250_r1';out.mkdir(exist_ok=False)
    for name in hashes:(out/('source_'+name)).write_bytes((HERE/name).read_bytes())
    save(out/'protocol.json',dict(planned=250,scope='CPU_initial_catalog_comparison_not_LLM_acceptance_or_threat',
        source_sha256=hashes,protected_source_sha256=protected,config_sha256=sha(cfg_path),
        original_static_stage_sha256=sha(old/'all250_r1/stage_result.json'),
        original_union_sha256=sha(old/'union250_r1/stage_result.json'),
        inherited_dynamic_scope='reuse_previous_exact_factory_dynamic_catalogs_where_original_static_absent; unchanged_algorithms',
        original_forward_minimum_m=15,new_route_footprint_entry_minimum_m=15,
        new_initial_ego_OBB_margin_m=.5,background_and_area_validators_unchanged=True,
        new_API_requests=0,new_execution_frames=0,D=None,goal_complete=False))
    rows=[];checks=[];started=time.time()
    try:
        for index,(filename,digest) in enumerate(zip(manifest['scenario_files'],manifest['scenario_sha256'])):
            if index%10==0:
                check=verify_sources(protocol,protected);checks.append(check)
                save(out/'runtime_source_integrity.json',checks)
                if not check['verified']:raise ValueError('protected_algorithm_changed')
                if shutil.disk_usage(ROOT).free<5.5*2**30:raise RuntimeError('insufficient_disk_space')
            if any(sha(HERE/name)!=h for name,h in hashes.items()):raise ValueError('route_audit_code_changed')
            assert sha(filename)==digest
            baseline=json.loads((old/f'all250_r1/scene{index:03d}.json').read_text())
            geometry=pickle.loads(Path(filename).read_bytes());geometry['_audit_filename']=Path(filename).name
            env=initial_environment(geometry,cfg)
            planner=LLMAdversarialPlanner(model_name='qwen3.8-2.4t-a95b',model_names=['qwen3.8-2.4t-a95b'],provider='dashscope',client=NS())
            planner.use_obstacles=True
            builder=CandidateBuilder(cfg.sim.ego_profile)
            with install(planner,env,enabled=True,initial_margin=builder.margin) as receipt:
                catalog,audit=build_static_combinations(env,planner,builder)
            old_union=previous['rows'][index]
            assert old_union['scene']==index
            # 原静态有目录的场景不需要重建动态；可用性的旧证据完整保留。
            union_available=bool(catalog) or old_union['candidate_union_available']
            row=dict(scene=index,scenario_sha256=digest,old_static_available=baseline['catalog_available'],
                old_static_count=baseline['catalog_count'],new_static_available=bool(catalog),
                new_static_count=len(catalog),old_union_available=old_union['candidate_union_available'],
                new_union_available=union_available,static_catalog=catalog,static_audit=audit,
                route_fallback_events=receipt['events'],accepted_LLM_plan=None,actual_execution_frames=0,D=None)
            # 不允许将“保留原成功路径”只写在说明中，逐场检查可用性没有回退。
            if row['old_static_available'] and not row['new_static_available']:
                raise AssertionError('old_static_availability_regressed')
            rows.append(row);save(out/f'scene{index:03d}.json',row)
            save(out/'progress.json',dict(planned=250,scanned=len(rows),elapsed_seconds=time.time()-started))
        final=verify_sources(protocol,protected);checks.append(final);save(out/'runtime_source_integrity.json',checks)
        if not final['verified']:raise ValueError('protected_algorithm_changed_at_end')
        save(out/'stage_result.json',dict(status='completed',planned=250,scanned=len(rows),
            old_static_available=sum(r['old_static_available'] for r in rows),
            new_static_available=sum(r['new_static_available'] for r in rows),
            old_union_available=sum(r['old_union_available'] for r in rows),
            new_union_available=sum(r['new_union_available'] for r in rows),
            recovered_union_scenes=[r['scene'] for r in rows if not r['old_union_available'] and r['new_union_available']],
            still_empty_scenes=[r['scene'] for r in rows if not r['new_union_available']],
            static_availability_regressions=[r['scene'] for r in rows if r['old_static_available'] and not r['new_static_available']],
            protected_algorithm_unchanged=True,elapsed_seconds=time.time()-started,
            actual_LLM_acceptance=None,new_API_requests=0,new_execution_frames=0,D=None,goal_complete=False))
    except Exception as error:
        save(out/'stage_result.json',dict(status='failed',planned=250,scanned=len(rows),unattempted=250-len(rows),
            error_type=type(error).__name__,error=str(error),new_API_requests=0,D=None,goal_complete=False))
        raise


if __name__=='__main__':main()
