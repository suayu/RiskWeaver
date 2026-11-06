"""效果统计排除系统故障；完整台账和共同场景配对统计并列输出。"""
import math
from scene_fault_policy import is_fault


def interval(success,n):
    # Wilson区间只描述小样本不确定性，不证明场景抽样具有总体代表性。
    if not n:return None
    z=1.959963984540054;p=success/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]


def summarize(planned,attempts,reports):
    ids=[r['scenario_index'] for r in attempts]
    if len(ids)!=len(set(ids)) or not set(ids)<=set(planned):raise ValueError('duplicate_or_unknown_attempt')
    usable=[r for r in attempts if not is_fault(r)];n=len(usable)
    success=sum(reports.get(r['scenario_index'],{}).get('effective_danger_constructed') is True for r in usable)
    def mean(key):
        values=[r[key] for r in usable if isinstance(r.get(key),(int,float)) and math.isfinite(r[key])]
        return dict(mean=sum(values)/len(values) if values else None,observed=len(values),denominator=n)
    return dict(planned=len(planned),attempted=len(attempts),fault_count=len(attempts)-n,
        excluded_fault_scenes=[r['scenario_index'] for r in attempts if is_fault(r)],
        unattempted=[s for s in planned if s not in ids],effective_denominator=n,
        effective_danger_count=success,effective_danger_rate=success/n if n else None,
        effective_danger_Wilson95=interval(success,n),
        target_rate_90pct_observed=success/n>=.9 if n else None,
        collision_count=sum(r.get('collision') is True for r in usable),
        collision_rate=sum(r.get('collision') is True for r in usable)/n if n else None,
        scores={k:mean(k) for k in ['scene_challenge_score','outcome_severity_score','autonomous_driving_ability_score','min_ttc_s','progress']},
        sampling_warning='fixed20; fault_exclusion_can_be_nonrandom; observed_rate_not_population_guarantee')


def paired(left,right,reports_left,reports_right):
    a={r['scenario_index']:r for r in left if not is_fault(r)}
    b={r['scenario_index']:r for r in right if not is_fault(r)}
    common=sorted(a.keys()&b.keys());rows=[]
    for scene in common:
        l=reports_left.get(scene,{}).get('effective_danger_constructed') is True
        r=reports_right.get(scene,{}).get('effective_danger_constructed') is True
        rows.append(dict(scene=scene,old_effective=l,skills_effective=r,difference=int(r)-int(l)))
    return dict(common_nonfault_scenes=common,n=len(common),rows=rows,
        mean_coverage_difference=sum(r['difference'] for r in rows)/len(rows) if rows else None,
        skills_only_success=sum(r['difference']==1 for r in rows),old_only_success=sum(r['difference']==-1 for r in rows),
        scope='same_scenes_pairing; excludes_faults_in_either_arm; not_skill_text_only_causality')
