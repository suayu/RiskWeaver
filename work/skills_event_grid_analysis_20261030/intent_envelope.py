"""只读的高层锚点必要可达性与源CV预览；不作为新增运行门控。"""
import numpy as np


def acceleration_envelope(actor,anchors,max_acceleration=6.):
    """若||a||≤A，则||p(t)-p0-v0*t||≤A*t²/2；通过仅为必要条件。"""
    state=np.asarray(actor,float);points=np.asarray(anchors,float)
    if state.ndim!=1 or len(state)<4 or points.shape!=(4,2) or not np.isfinite(points).all() or not np.isfinite(state[:4]).all() or max_acceleration<=0:
        raise ValueError('invalid_necessary_envelope_input')
    t=np.arange(1.,4.);reference=state[:2]+t[:,None]*state[2:4]
    residual=np.linalg.norm(points[1:]-reference,axis=1);bound=.5*max_acceleration*t*t
    return dict(times_seconds=t.tolist(),constant_velocity_residual_m=residual.tolist(),necessary_max_residual_m=bound.tolist(),
        signed_excess_m=(residual-bound).tolist(),max_signed_excess_m=float(np.max(residual-bound)),
        necessary_exact_anchor_condition_pass=bool(np.all(residual<=bound+1e-6)),numerical_tolerance_m=1e-6,
        initial_anchor_position_difference_m=float(np.linalg.norm(points[0]-state[:2])),max_acceleration_mps2=max_acceleration,
        sufficient_dynamic_or_background_or_ego_area_certificate=False,new_runtime_gate=False)


def analyze(root,output):
    from pathlib import Path
    import hashlib,json,sys
    project=Path('/home2/zhaoyx/scenario-dreamer');sys.path[:0]=[str(project/'work/skills_temporal_candidates_20261029'),str(project/'work/skills_incremental_evidence_20261025')]
    from temporal_intents import fractional_minimum
    from incremental_evidence import incremental_card
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError('preserve_previous_intent_audit')
    result=json.loads((root/'results.json').read_text());samples=json.loads((root/'inputs.json').read_text());by_scene={s['scene']:s for s in samples}
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(root/'resolved_config.yaml');limit=float(cfg.sim.traffic_model.guidance.loss_configs.scenario_collision.max_acceleration)
    records=[];sources={}
    for row in result['results']:
        sample=by_scene[row['scene']];plan=row.get('plan');record=dict(scene=row['scene'],mode=row['mode'],module_accepted=row['accepted_plan'],
            necessary_anchor_condition=None,source_CV_proxy=None)
        file=root/f'scene{row["scene"]:03d}_{row["mode"]}/result.json';sources[str(file.relative_to(project))]=hashlib.sha256(file.read_bytes()).hexdigest()
        if row['accepted_plan']:
            target=plan['attack_target_id'];frame=sample['frame'];actor=np.asarray(frame['agents'])[target]
            candidate=dict(candidate_id=plan.get('candidate_id') or f'legacy_scene{row["scene"]}',target_id=target,anchors=plan['anchors'],duration=10)
            record['necessary_anchor_condition']=acceleration_envelope(actor,plan['anchors'],limit)
            try:incremental=incremental_card(sample['state'],candidate,.1,1.)
            except ValueError as error:incremental=dict(valid=False,reason=str(error),proxy_not_filled_with_zero=True)
            record['source_CV_proxy']=dict(active=fractional_minimum(plan['anchors'],frame['ego'],.1,1.),forecast=fractional_minimum(plan['anchors'],frame['ego'],.1,3.),
                incremental=incremental)
        records.append(record)
    report=dict(records=records,input_sha256=sources,planned_scenes_per_arm=10,all_failed_unattempted_in_primary_denominator=True,
        scope='necessary_exact_waypoint_condition_and_source_CV_proxy_only; no_new_gate_or_execution_or_causal_risk_claim',new_API_calls=0,new_execution_frames=0,D=None)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();r=analyze(args.root,args.output)
    print('audited',len(r['records']))
