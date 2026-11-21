"""共享GPU只改变资源准入；不改变扩散样本、优化器或场景安全约束。"""
import math


def eligible_shared(gpu_rows, cap_gib=14., reserve_gib=3., allowed=('0','1'), reserved=()):
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in [cap_gib,reserve_gib]):
        raise ValueError('invalid_shared_memory_policy')
    results=[]
    for line in gpu_rows.splitlines():
        if not line.strip():continue
        index,total,free,util=[x.strip() for x in line.split(',')]
        total,free,util=float(total),float(free),float(util)
        if index in allowed and index not in reserved and free >= (cap_gib+reserve_gib)*1024 and cap_gib*1024<total:
            results.append(dict(index=index,total_MiB=total,free_MiB=free,util_percent=util))
    return sorted(results,key=lambda x:(-x['free_MiB'],x['index']))


def memory_fraction(cap_gib,total_bytes):
    if isinstance(cap_gib,bool) or not math.isfinite(cap_gib) or not 0<cap_gib*2**30<total_bytes:
        raise ValueError('memory_cap_must_be_below_device_capacity')
    return cap_gib*2**30/total_bytes


def fixture_response(payload):
    """资源夹具在当前目录内作确定性选择；不模拟新鲜LLM调用或输出连续轨迹。"""
    candidates=payload['context']['candidates']
    if not candidates:raise ValueError('resource_fixture_requires_current_catalog')
    evidence=payload['context']['priority_evidence']
    def score(candidate):
        cid=candidate['candidate_id'];card=evidence[cid]['facts'][cid+':contrast']
        return (-card['min_gap_difference_m'],card['candidate_min_gap_m'],cid)
    chosen=min(candidates,key=score);cid=chosen['candidate_id'];facts=evidence[cid]
    return dict(request_id=payload['request_id'],state_hash=payload['state_hash'],catalog_hash=payload['catalog_hash'],
        ranking=[cid],scene_analysis='Deterministic resource fixture, not an LLM response or attack-performance evidence.',
        candidate_reasons={cid:'Resource-only catalog fixture; existing semantic and joint safety checks remain mandatory.'},
        no_attack_reason='',grounding={cid:dict(target_id=chosen['target_id'],target_role='background_vehicle',
        active_horizon_seconds=facts['active_horizon_seconds'],evidence_ids=[cid+':actor',cid+':validity',cid+':active',cid+':contrast'],
        interaction_basis='exploratory_probe',contrast_direction=facts['facts'][cid+':contrast']['proxy_direction'])})
