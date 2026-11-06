"""按计划记录攻击中的响应空间；不把面积直接解释成反应时间或可避碰证明。"""
import math


def summarize_attacks(records):
    """区分攻击执行时间曲线与每帧预测时域曲线；缺失测量保留null。"""
    groups = {}
    for record in records:
        attack = record.get('attack')
        if not attack:
            continue
        key = (attack.get('source_step'), attack.get('target_id'), attack.get('strategy'))
        group = groups.setdefault(key, {'attack':attack, 'area_over_attack_frames':[],
                                         'forecast_area_curves':[]})
        reach = record.get('reachability')
        valid = bool(record.get('valid') and reach)
        group['area_over_attack_frames'].append({
            'source_step':None if reach is None else reach['source_step'],
            'area_m2':reach['dangerous_area_m2'] if valid else None,
            'original_area_m2':reach['original_area_m2'] if valid else None,
            'valid':valid, 'measurement_status':record.get('measurement_status')})
        if reach:
            group['forecast_area_curves'].append({'source_step':reach['source_step'],
                'baseline':reach.get('original_area_curve',[]),
                'attack':reach.get('dangerous_area_curve',[]),
                'minimum_future_area_m2':reach.get('minimum_dangerous_area_m2')})
    for group in groups.values():
        values = [r['area_m2'] for r in group['area_over_attack_frames'] if r['valid']]
        future = [c['minimum_future_area_m2'] for c in group['forecast_area_curves']
                  if c['minimum_future_area_m2'] is not None]
        group['minimum_terminal_area_during_attack_m2'] = min(values) if values else None
        group['minimum_future_area_during_attack_m2'] = min(future) if future else None
        group['valid_frame_count'] = len(values)
        group['observed_frame_count'] = len(group['area_over_attack_frames'])
        group['interpretation'] = 'theoretical_response_space; not_evasion_witness_or_reaction_time'
    return list(groups.values())
