"""模型池配对的只读验收；逻辑请求与真实传输分开，失败留在预定分母。"""
import argparse
from collections import Counter
import json
import hashlib
import math
from pathlib import Path
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path.insert(0,str(ROOT/'work/skills_audit_v2_dev_20261020'))
from audit_enhanced import enhanced_audit


def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def qualify_scene(attempt,ledger,requests,budget=4):
    """一次生成失败可与其他场景的成功共存，但失败场景自身绝不算有效攻击。"""
    reasons=[]
    if attempt.get('status')!='completed' or attempt.get('executed_steps',0)<=0:reasons.append('episode_not_completed')
    if attempt.get('generation_failure'):reasons.append('generation_failure')
    if attempt.get('runtime_failure'):reasons.append('runtime_failure')
    if attempt.get('planner_service_failures',0):reasons.append('planner_service_failure')
    if attempt.get('background_collision_frames',0) or attempt.get('background_static_collision_frames',0):reasons.append('background_collision')
    if not finite(attempt.get('D')) or attempt.get('D_valid_frames',0)<=0:reasons.append('invalid_D')
    complete=attempt.get('completeness',{})
    if not all(complete.get(k,False) for k in ['trace_exists','terminal_state_present','state_sequence_complete','video_frames_match']) or complete.get('malformed_trace_lines',0):reasons.append('artifact_incomplete')
    if ledger.get('executed_plans',0)<1 or attempt.get('actual_attack_frames',0)<1:reasons.append('no_executed_plan')
    if any(ledger.get(k) for k in ['unmatched_execution_steps','execution_without_anchor_guidance_steps','duplicate_execution_markers']) or any(p.get('exceeds_declared_duration') for p in ledger.get('plans',[])):reasons.append('execution_attribution_invalid')
    if len(requests)>budget:reasons.append('physical_request_budget_exceeded')
    # 固定计划重放可以验证控制，却不属于新鲜LLM生成成功，必须有真实响应记录。
    if not any(not r.get('error_type') and r.get('model_returned') for r in requests):reasons.append('no_fresh_model_response')
    return dict(scenario_index=attempt['scenario_index'],qualified=not reasons,reasons=reasons,
                physical_requests=len(requests),actual_attack_frames=attempt.get('actual_attack_frames',0),
                D=attempt.get('D'),D_valid_frames=attempt.get('D_valid_frames',0))


def paired_gate(groups,threshold=6):
    old=groups['legacy_shared_recovery'];new=groups['skills_full_shared_recovery']
    passed=old['group_integrity_pass'] and new['group_integrity_pass'] and new['qualified_scenes']>=threshold and new['qualified_scenes']>=old['qualified_scenes']
    return dict(passed=bool(passed),required_qualified_scenes=threshold,
                legacy_qualified=old['qualified_scenes'],skills_qualified=new['qualified_scenes'],
                scope='prospective development-to-validation gate; not universal non-inferiority, skill-text causality or goal completion')


def audit_pool(root):
    base=enhanced_audit(root);manifest=json.loads((root/'manifest.json').read_text())
    launch=json.loads((root/'launch.json').read_text());budget=launch.get('max_api_calls_per_scene',4)
    expected=manifest['original_scenario_indices'];result={}
    source_hash_path=root/'source_sha256.json';receipt_path=root/'runtime_source_integrity.json'
    hashes=json.loads(source_hash_path.read_text()) if source_hash_path.exists() else {}
    receipts=json.loads(receipt_path.read_text()).get('checks',[]) if receipt_path.exists() else []
    snapshot_bad=[n for n,h in hashes.items() if not (root/'source_snapshot'/n).is_file()
                  or hashlib.sha256((root/'source_snapshot'/n).read_bytes()).hexdigest()!=h]
    source_ok=bool(hashes and not snapshot_bad and receipts and all(x.get('verified') for x in receipts))
    for arm,g in base['groups'].items():
        log=root/f'{arm}_api_calls.jsonl';requests=[];malformed=0
        if log.exists():
            for line in log.read_text().splitlines():
                try:requests.append(json.loads(line))
                except json.JSONDecodeError:malformed+=1
        by_scene={index:[r for r in requests if r.get('scenario_index')==index] for index in expected}
        unknown=[r.get('scenario_index') for r in requests if r.get('scenario_index') not in expected]
        ledger={x['scenario_index']:x for x in g['plan_execution_ledgers']}
        qualified=[qualify_scene(a,ledger.get(a['scenario_index'],{}),by_scene.get(a['scenario_index'],[]),budget) for a in g['attempts']]
        done={x['scenario_index'] for x in qualified}
        qualified.extend(dict(scenario_index=i,qualified=False,reasons=['unattempted'],physical_requests=0,
                              actual_attack_frames=0,D=None,D_valid_frames=0) for i in expected if i not in done)
        group_protocol=root/arm/'modelpool_budget_protocol.json'
        p=json.loads(group_protocol.read_text()) if group_protocol.exists() else {}
        cfg_ok=False
        config=root/arm/'resolved_config.yaml'
        if config.exists():
            from omegaconf import OmegaConf
            cfg=OmegaConf.load(config)
            cfg_ok=(cfg.sim.steps==launch['max_frames']==400 and cfg.sim.llm.provider=='dashscope'
                    and cfg.sim.traffic_model.guidance.params.inner_lr==.2
                    and cfg.sim.traffic_model.guidance.params.inner_beta==.5
                    and cfg.sim.traffic_model.guidance.params.n_guide_steps==2)
        integrity=bool(g['all_planned_attempted'] and g['artifacts_complete'] and g['exact_manifest_coverage']['verified']
            and source_ok and all(any(x.get('arm')==arm and x.get('phase')==phase for x in receipts) for phase in ['before','after'])
            and not base['frozen_input_hash_failures'] and not unknown and not malformed
            and not (g['runtime_failures'] or g['planner_service_failures'] or g['background_collision_frames'])
            and all(len(v)<=budget for v in by_scene.values()) and cfg_ok
            and p.get('model_source_integrity',{}).get('verified') and p.get('free_protection_operator_confirmed')
            and p.get('transport_limit')==budget)
        usage=[r.get('usage') for r in requests if not r.get('error_type')]
        tokens_known=all(isinstance(u,dict) and isinstance(u.get('total_tokens'),int) for u in usage)
        token_sum=sum(u['total_tokens'] for u in usage) if tokens_known else None
        count=sum(q['qualified'] for q in qualified)
        result[arm]=dict(planned_scenes=len(expected),attempted_scenes=g['attempted_scenes'],qualified_scenes=count,
            qualified_coverage=count/len(expected),group_integrity_pass=integrity,scene_qualification=qualified,
            physical_requests=len(requests),planning_opportunities=sum(a['planning_queries'] for a in g['attempts']),
            reported_logical_or_legacy_api_count=g['api_calls'],physical_and_logical_counts_need_not_match=True,
            model_distribution=g['transport']['model_distribution'],total_tokens=token_sum,
            generation_failure_count=g['generation_failures'],runtime_failure_count=g['runtime_failures'],
            planner_service_failure_count=g['planner_service_failures'],background_collision_frames=g['background_collision_frames'],
            D_mean_valid_scenes=g['D_mean_valid_scenes'],D_frame_coverage=g['D_frame_coverage'],
            metric_attempts=[{k:a.get(k) for k in ['scenario_index','executed_steps','collision','completed','off_route','progress','D','EA','OBB_TTC','actual_attack_frames','first_forecasts']} for a in g['attempts']],
            trajectory_realism=None,trajectory_realism_status='pending_comparison_with_real_reference_distribution',
            frozen_configuration_verified=cfg_ok,source_protocol=p)
    gate=paired_gate(result)
    return dict(experiment=str(root),groups=result,development_gate=gate,
                source_integrity=dict(verified=source_ok,snapshot_hash_failures=snapshot_bad,receipts=receipts),
                goal_complete=False,model_mixture_scope='operational common-pool comparison; model confounding must be reported',
                existing_single_model_audit=base)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.experiment.resolve() in args.output.resolve().parents:raise ValueError('audit must not write immutable experiment inputs')
    result=audit_pool(args.experiment);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result['development_gate'],ensure_ascii=False))


if __name__=='__main__':main()
