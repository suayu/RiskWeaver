"""当前状态的确定性事实卡；只核验引用主张，不改变候选或安全门槛。"""
import math

FIELDS = {'scene_fact_id', 'source_step', 'longitudinal_relation', 'lateral_relation', 'heading_relation'}


def relation_card(ego, actor, target_id, source_step):
    """自车右/前坐标由姿态计算；航向分界与既有active-window配方一致。"""
    if isinstance(target_id, bool) or not isinstance(target_id, int):
        raise ValueError('scene_fact_target_identity')
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step < 0:
        raise ValueError('scene_fact_source_time')
    if len(ego) < 5 or len(actor) < 5 or not all(math.isfinite(float(v)) for v in list(ego[:5])+list(actor[:5])):
        raise ValueError('scene_fact_missing_or_nonfinite_state')
    dx,dy = float(actor[0]-ego[0]),float(actor[1]-ego[1])
    yaw = float(ego[4])
    right = dx*math.sin(yaw)-dy*math.cos(yaw)
    forward = dx*math.cos(yaw)+dy*math.sin(yaw)
    angle = abs(math.atan2(math.sin(float(actor[4])-yaw), math.cos(float(actor[4])-yaw)))
    # 0.5米仅表示位置标签的对齐带，绝不是新攻击者距离门槛。
    band=.5
    return dict(fact_id=f'actor{target_id}@step{source_step}:pose', target_id=target_id,
        source_step=source_step,role='background_vehicle',coordinate_frame='ego_right_forward',
        right_m=right,forward_m=forward,heading_difference_degrees=math.degrees(angle),
        longitudinal_relation='ahead' if forward>band else 'behind' if forward< -band else 'aligned',
        lateral_relation='right' if right>band else 'left' if right< -band else 'aligned',
        heading_relation='opposing_headings' if angle>2.4 else 'similar_headings' if angle<.7 else 'crossing_headings',
        position_alignment_band_m=band,heading_boundaries_radians=[.7,2.4],
        scope='current_pose_and_kinematic_heading; not_map_lane_topology_or_tactical_correctness',
        executed=False,actual_threat_certified=False)


def build_cards(state, candidates, source_step):
    actors={}
    for row in state['agents']:
        ident=row['id']
        if isinstance(ident,bool) or not isinstance(ident,int) or ident in actors:
            raise ValueError('scene_fact_ambiguous_actor')
        actors[ident]=row['state']
    cards={}
    for candidate in candidates:
        cid=candidate['candidate_id'];target=candidate['target_id']
        if cid in cards or isinstance(target,bool) or not isinstance(target,int) or target not in actors:
            raise ValueError('scene_fact_missing_or_duplicate_identity')
        cards[cid]=relation_card(state['ego_state'],actors[target],target,source_step)
    return cards


def validate_claims(response, cards):
    """逐个排序ID核对新增字段，不自动修正模型原文，也不接受缺失/陈旧身份。"""
    if not isinstance(response,dict) or not isinstance(response.get('ranking'),list) or not isinstance(response.get('grounding'),dict):
        raise ValueError('scene_fact_response_schema')
    if set(response['grounding'])!=set(response['ranking']):
        raise ValueError('scene_fact_grounding_keys')
    for cid in response['ranking']:
        if cid not in cards:
            raise ValueError('scene_fact_unknown_candidate')
        row=response['grounding'][cid];card=cards[cid]
        if not isinstance(row,dict) or not FIELDS <= set(row):
            raise ValueError('scene_fact_required_claim_missing')
        if row['scene_fact_id']!=card['fact_id']:
            raise ValueError('scene_fact_reference_mismatch')
        if isinstance(row['source_step'],bool) or not isinstance(row['source_step'],int) or row['source_step']!=card['source_step']:
            raise ValueError('scene_fact_source_time_mismatch')
        for name in ['longitudinal_relation','lateral_relation','heading_relation']:
            if row[name]!=card[name]:
                raise ValueError('scene_fact_'+name+'_mismatch')
