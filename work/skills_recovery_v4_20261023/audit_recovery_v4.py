"""新冻结阶段的只读审计：沿用原多数门槛，另核对v4组件与整数视频开关。"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_modelpool_budget_dev_20261022'))
from pool_acceptance import audit_pool as legacy_audit_pool,paired_gate


def load(path):return json.loads(path.read_text()) if path.exists() else {}


def component_configuration(cfg,recovery,video,arm,hashes):
    """缺失收据或任何非共同控制配置均不能靠旧审计的通过来覆盖。"""
    tail='on' if arm=='skills_full_shared_recovery' else 'off'
    return bool(cfg.sim.evaluation.reachability.get('execution_gate_policy')=='positive_area'
        and recovery.get('execution_area_policy')=='positive_area'
        and recovery.get('temporal_recovery')=='on'
        and recovery.get('temporal_max_consecutive')==1
        and recovery.get('temporal_acceptance_budget_seconds')==60.
        and recovery.get('route_tail_preflight')==tail
        and video.get('video_export')=='integer' and video.get('fps')==5
        and recovery.get('source_sha256') and video.get('source_sha256')
        and all(hashes.get(n)==h for p in (recovery,video) for n,h in p.get('source_sha256',{}).items()))


def audit_v4(root):
    result=legacy_audit_pool(root);result['pre_v4_gate']=result['development_gate']
    hashes=load(root/'source_sha256.json')
    for arm,group in result['groups'].items():
        from omegaconf import OmegaConf
        config_path=root/arm/'resolved_config.yaml'
        recovery=load(root/arm/'recovery_v4_protocol.json');video=load(root/arm/'video_export_protocol.json')
        flags_ok=config_path.exists() and component_configuration(OmegaConf.load(config_path),recovery,video,arm,hashes)
        counts=Counter();recovered=[];malformed=0
        for p in (root/arm).rglob('attempt_result.json'):
            a=load(p);rows=[]
            for line in p.with_name('execution_trace.jsonl').read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:malformed+=1
            completed={r['step'] for r in rows if r.get('kind')=='state' and r['step']>0}
            for row in rows:
                if row.get('kind')=='llm_output':
                    c=(row.get('planner_trace') or {}).get('context') or {}
                    if not isinstance(c,dict):continue
                    before=any(c.get(k) for k in ('candidates','obstacle_candidates','preflight_withheld_candidates',
                        'preflight_withheld_obstacles','route_tail_withheld_candidates','route_tail_withheld_obstacles'))
                    counts['catalog_queries']+=1;counts['nonempty_before_all_preflights']+=bool(before)
                    counts['nonempty_after_all_preflights']+=bool(c.get('candidates') or c.get('obstacle_candidates'))
                    counts['route_tail_declines']+=(c.get('route_tail_preflight') or {}).get('ready') is False
                if row.get('kind')=='execution_prediction':
                    m=((row.get('joint') or {}).get('metadata') or {}).get('temporal_joint_recovery')
                    if m:
                        recovered.append(dict(scene=a['scenario_index'],source_step=row['step'],
                                              actually_executed=row['step']+1 in completed,recovery=m))
        temporal_ok=all(x['recovery'].get('recovered') and x['recovery'].get('consecutive')==1
            and x['recovery'].get('all_active_agents_verified') and x['recovery'].get('no_extrapolation')
            and x['recovery'].get('remaining_steps',0)>=30
            and x['recovery'].get('additional_same_frame_diffusion_samples')==0
            and x['recovery'].get('additional_LLM_calls')==0
            and x['recovery'].get('validation_seconds',float('inf'))<=60.
            and x['recovery'].get('area_gate',{}).get('allowed') for x in recovered)
        group['v4_component_integrity_pass']=bool(flags_ok and temporal_ok and malformed==0)
        group['group_integrity_pass']&=group['v4_component_integrity_pass']
        group['v4_protocol']=dict(configuration_verified=bool(flags_ok),temporal_trace_valid=temporal_ok,
            recovery_forecasts=len(recovered),recovery_executed_frames=sum(x['actually_executed'] for x in recovered),
            recovered_forecasts=recovered,catalog_with_all_withheld_candidates=dict(counts),
            denominator_note='route-tail or measurement declines remain planning opportunities; failed and unattempted scenes remain in all10')
    result['development_gate']=paired_gate(result['groups'])
    result['v4_audit_scope']='prospective common execution-area policy/temporal recovery/integer video; skills-only scheduling; unchanged D and majority criteria'
    return result
