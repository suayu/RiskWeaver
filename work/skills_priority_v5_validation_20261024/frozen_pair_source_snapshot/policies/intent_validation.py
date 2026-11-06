"""候选发布与最终计划共用的快照语义预检；不代替连续执行门禁。"""
import hashlib
import json
import numpy as np


def canonical_hash(value):
    """哈希严格 JSON，用于阻止跨状态、跨目录的响应复用。"""
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                     allow_nan=False,separators=(',',':')).encode()).hexdigest()


def validate_world_intent(planner, state, candidate):
    """世界坐标候选仅转换一次，调用原验证器并返回具体数值证据。"""
    ego=np.asarray(state['ego_state'],float)
    anchors=np.asarray(candidate['anchors'],float)
    if anchors.shape!=(4,2) or not np.isfinite(anchors).all():
        return dict(accepted=False,reason='攻击轨迹必须包含四个有限二维锚点',metrics={})
    angle=np.pi/2-ego[4]
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    local=(anchors-ego[:2])@rotation.T
    target=next((a for a in state['agents'] if a['id']==candidate['target_id']),None)
    metrics={}
    if anchors.shape==(4,2) and np.isfinite(anchors).all() and target is not None:
        velocities=np.diff(anchors,axis=0)
        metrics=dict(initial_position_error_m=float(np.linalg.norm(anchors[0]-np.asarray(target['state'])[:2])),
            initial_velocity_delta_mps=float(np.linalg.norm(velocities[0]-np.asarray(target['state'])[2:4])),
            max_segment_speed_mps=float(np.max(np.linalg.norm(velocities,axis=1))),
            max_segment_acceleration_mps2=float(np.max(np.linalg.norm(np.diff(velocities,axis=0),axis=1))),
            aligned_ego_distance_m=np.linalg.norm(anchors[1:]-(ego[:2]+np.arange(1,4)[:,None]*ego[2:4]),axis=1).tolist(),
            terminal_lateral_m=float(local[-1,0]))
    plan=dict(attack=True,attack_target_id=candidate['target_id'],strategy=candidate['strategy'],anchors=local.tolist())
    try:
        if isinstance(candidate['target_id'],bool):
            raise ValueError('布尔值不能作为车辆ID')
        if isinstance(candidate['duration'],bool) or not isinstance(candidate['duration'],int) or not 1<=candidate['duration']<=400:
            raise ValueError('意图有效期超界')
        planner._validate_attack_plan(plan,planner._normalize_env_state(state))
    except (ValueError,TypeError,KeyError) as error:
        return dict(accepted=False,reason=str(error),metrics=metrics)
    return dict(accepted=True,reason=None,metrics=metrics)
