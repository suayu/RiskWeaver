"""静态候选的同口径排序事实；只提供提示，不排序候选或增加接受门控。"""
import hashlib
import json
import math
from numbers import Real


METRICS = (
    'geometric_free_area_m2',
    'background_CV_margin_m',
    'primitive_count',
)


def finite_number(value):
    """缺失、布尔值和非有限值保持未知，不能冒充零或无穷安全裕度。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value) if math.isfinite(float(value)) else None


def build_facts(catalog):
    """只在同一几何定义与分辨率的组内比较；同时显式列出并列排名。"""
    groups = {}
    seen = set()
    for item in catalog:
        cid = item['obstacle_candidate_id']
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError('invalid_or_duplicate_static_candidate_id')
        seen.add(cid)
        basis = item.get('geometric_area_basis')
        resolution = finite_number(item.get('area_resolution_m'))
        # 不知道面积口径的条目单独成组，防止把不同空间窗口混成一个排行榜。
        comparable = isinstance(basis, str) and bool(basis) and resolution is not None and resolution > 0
        key = (basis, resolution, None if comparable else cid)
        groups.setdefault(key, []).append(item)
    output = []
    for (basis, resolution, isolated), items in groups.items():
        metrics = {}
        for name in METRICS:
            values = {item['obstacle_candidate_id']: finite_number(item.get(name)) for item in items}
            valid = {cid: value for cid, value in values.items() if value is not None}
            rows = {}
            for cid, value in values.items():
                if value is None:
                    rows[cid] = dict(value=None, status='unknown_not_zero_or_infinite')
                    continue
                less = [other for other, x in valid.items() if x < value]
                greater = [other for other, x in valid.items() if x > value]
                equal = [other for other, x in valid.items() if x == value]
                rows[cid] = dict(value=value, status='finite', smaller_ids=less,
                    larger_ids=greater, equal_value_ids=equal,
                    competition_rank_ascending=1 + len(less),
                    dense_rank_ascending=1 + len({valid[x] for x in less}),
                    competition_rank_descending=1 + len(greater),
                    dense_rank_descending=1 + len({valid[x] for x in greater}))
            metrics[name] = dict(valid_count=len(valid), total_count=len(items),
                all_values_known=len(valid) == len(items),
                minimum_known_ids=[cid for cid, value in valid.items() if value == min(valid.values())] if valid else [],
                maximum_known_ids=[cid for cid, value in valid.items() if value == max(valid.values())] if valid else [],
                candidates=rows)
        output.append(dict(area_basis=basis if isinstance(basis, str) else None,
            area_resolution_m=resolution, comparable_scope_known=isolated is None,
            candidate_ids=[item['obstacle_candidate_id'] for item in items], metrics=metrics))
    facts = dict(version='static_metric_facts_prototype_v1', groups=output,
        comparison_scope='within_each_published_group_only',
        input_candidate_order_unchanged=True, no_new_acceptance_gate=True,
        area_is_not_dynamic_reachability_or_escape=True,
        area_is_not_lane_blockage_fraction_or_threat=True,
        no_baseline_area_or_blockage_percentage_inferred=True,
        primitive_count_is_not_attack_strength=True,
        unknown_margin_is_not_zero_infinite_or_safety_certificate=True,
        rank_definition='competition=1+number_of_strictly_smaller_candidates; dense=1+number_of_distinct_smaller_values; ties_explicit; exact_published_values')
    # 摘要只覆盖上述派生事实，不声称替代完整输入/实验的源码与数据哈希。
    facts['derived_facts_sha256'] = hashlib.sha256(json.dumps(facts, sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    return facts
