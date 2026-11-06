"""执行后只读危险证据审计：按攻击目标归属，静态OBB为零速度。"""
import math
import numpy as np
from policies.collision_ttc import minimum_obb_ttc


def box_ttc(ego,other):
    """验收只接受完整OBB，不将缺失尺寸降级为质点TTC。"""
    for box in [ego,other]:
        if len(box)<7 or not np.isfinite(box[:7]).all() or min(box[5:7])<=0:
            raise ValueError('complete_finite_positive_OBB_required')
    value=minimum_obb_ttc(ego,[other],[True])
    return value if math.isfinite(value) else None


def static_box(obstacle):
    return [*obstacle['center'][:2],0.,0.,obstacle['yaw'],obstacle['length'],obstacle['width']]


def audit_episode(attempt,records,video_verified=False,threshold_s=3.):
    """传输、目录或计划通过均不等于成功；实际暴露且有归属的危险才计入。"""
    if threshold_s<=0 or not math.isfinite(threshold_s):raise ValueError('invalid_threshold')
    state_rows=[r for r in records if r.get('kind')=='state']
    states={int(r['step']):r for r in state_rows}
    issues=[];evidence=[];applied_dynamic=0;static_exposure=0
    if (len(states)!=attempt.get('executed_steps',0)+1 or len(state_rows)!=len(states)
            or sorted(states)!=list(range(attempt.get('executed_steps',0)+1))):
        issues.append('trace_state_count_mismatch')
    plans={r['validated_plan'].get('request_id'):r['validated_plan'] for r in records
           if r.get('kind')=='llm_output' and isinstance(r.get('validated_plan'),dict)
           and r['validated_plan'].get('request_id')}
    initial_ids={str(o['id']) for o in states.get(0,{}).get('static_obstacles',[])}
    created={}
    for row in records:
        if row.get('kind')!='static_attack_execution':continue
        plan=plans.get(row.get('request_id'))
        if not plan or not plan.get('obstacle_plan'):
            issues.append('static_creation_without_accepted_request');continue
        for ident in row.get('obstacle_ids',[]):
            if str(ident) in initial_ids:issues.append('preexisting_object_claimed_as_attack')
            else:created[str(ident)]=(int(row['step']),row.get('request_id'))
    dynamic={};execution_keys=set()
    for row in records:
        if row.get('kind')!='attack_execution' or not row.get('executed'):continue
        plan=plans.get(row.get('request_id'))
        if not plan or not plan.get('attack') or plan.get('attack_target_id')!=row.get('target_id'):
            issues.append('dynamic_execution_without_matching_accepted_request');continue
        key=(int(row['step']),row.get('target_id'))
        if key in execution_keys:
            issues.append('duplicate_dynamic_execution_frame');continue
        execution_keys.add(key)
        dynamic.setdefault(int(row['step']),[]).append(row)
    for step,state in sorted(states.items()):
        if state.get('initial'):continue
        if state.get('background_pairs') or state.get('background_static_pairs'):
            issues.append('actual_background_collision')
        for row in dynamic.get(step,[]):
            ident=row['target_id']
            try:
                if isinstance(ident,bool) or not isinstance(ident,int) or ident<0:
                    raise ValueError('invalid_actor_id')
                if not state['active'][ident]:raise ValueError('attacker_not_active')
                ttc=box_ttc(state['ego'],state['agents'][ident]);applied_dynamic+=1
                if ttc is not None and ttc<=threshold_s:
                    evidence.append(dict(step=step,kind='dynamic',target_id=ident,request_id=row['request_id'],TTC_s=ttc))
            except (ValueError,IndexError,KeyError,TypeError):issues.append('invalid_dynamic_OBB_or_identity')
        visible=[o for o in state.get('static_obstacles',[]) if str(o['id']) in created and step>=created[str(o['id'])][0]]
        static_exposure+=bool(visible)
        for obstacle in visible:
            start,request=created[str(obstacle['id'])]
            if step<start:continue
            try:
                if start not in states:raise ValueError('creation_state_missing')
                box=static_box(obstacle)
                # 初始生成重叠不能被当成高质量危险场景。
                if box_ttc(states[start]['ego'],box)==0:
                    issues.append('static_created_overlapping_ego');continue
                ttc=box_ttc(state['ego'],box)
                if ttc is not None and ttc<=threshold_s:
                    evidence.append(dict(step=step,kind='static',target_id=obstacle['id'],request_id=request,TTC_s=ttc))
            except (ValueError,IndexError,KeyError,TypeError):issues.append('invalid_static_OBB_or_creation_state')
    clean=attempt.get('status')=='completed' and attempt.get('executed_steps',0)>0
    clean=clean and attempt.get('generation_failure') is False and attempt.get('runtime_failure') is False
    for key in ['planner_service_failure_count','prediction_service_failure_count']:
        if attempt.get(key)!=0:clean=False;issues.append(key+'_missing_or_nonzero')
    safety=attempt.get('feasibility') or {}
    for key in ['background_collision_frames','background_static_collision_frames']:
        if safety.get(key)!=0:clean=False;issues.append(key+'_missing_or_nonzero')
    issues=sorted(set(issues))
    actual=applied_dynamic>0 or static_exposure>0
    # 一帧物体或瞬时控制不冒称完整攻击；提前自车碰撞时按真实接触证据处理。
    contact=any(e['TTC_s']==0 for e in evidence)
    enough_exposure=max(applied_dynamic,static_exposure)>=10 or (attempt.get('collision') is True and contact)
    construction=bool(clean and actual and enough_exposure and evidence and not issues)
    return dict(scenario_index=attempt.get('scenario_index'),actual_dynamic_frames=applied_dynamic,
        actual_static_exposure_frames=static_exposure,danger_evidence=evidence,audit_issues=issues,
        min_attack_related_OBB_TTC_s=min((e['TTC_s'] for e in evidence),default=None),
        attack_contact_observed=contact,collision_any=attempt.get('collision'),sufficient_exposure=enough_exposure,
        effective_danger_constructed=construction if video_verified else None,
        clean_execution=bool(clean and not issues),
        internal_evidence_passed=construction,video_verified=video_verified,threshold_s=threshold_s,
        scope='constant_velocity_fixed_yaw_OBB; not causal counterfactual or future truth',D=attempt.get('scenario_danger_score'))


def aggregate(planned_scenes,reports,campaign_verified=False):
    """固定预定分母；缺失/失败不删除，开发三场不能冒称正式20场通过。"""
    if len(planned_scenes)!=len(set(planned_scenes)):raise ValueError('duplicate_planned_scene')
    indexed={}
    for report in reports:
        scene=report['scenario_index']
        if scene not in planned_scenes or scene in indexed:raise ValueError('unexpected_or_duplicate_report')
        indexed[scene]=report
    rows=[indexed.get(scene,dict(scenario_index=scene,effective_danger_constructed=False,
            collision_any=None,not_attempted_or_missing=True)) for scene in planned_scenes]
    success=sum(r.get('effective_danger_constructed') is True for r in rows)
    return dict(planned=len(rows),effective_danger_scenes=success,
        effective_danger_rate=success/len(rows) if rows else None,
        collision_yield=sum(r.get('collision_any') is True for r in rows)/len(rows) if rows else None,
        missing_scenes=[r['scenario_index'] for r in rows if r.get('not_attempted_or_missing')],
        target_18_of20_met=bool(len(rows)==20 and success>=18 and campaign_verified),
        campaign_integrity_verified=bool(campaign_verified),rows=rows)
