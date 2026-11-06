"""为可选技能链提供当前方位和攻击时钟事实；不修改候选、回复或接收门控。"""
import copy
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path


VERSION = 'source-fact-contract-v1-20261101'
FIELD = 'source_facts_v1'
RULE = 'rw-source-facts/SKILL.md'


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def lease_clock(previous_status, step, dt, scene_id=None):
    """以整数帧差区分开始年龄与到期年龄；历史意图接收不等于真实执行。"""
    unavailable = dict(available=False, accepted_intent_is_not_execution=True)
    if not _integer(step) or step < 0 or not _finite(dt) or dt <= 0:
        return dict(unavailable, reason='invalid_current_clock')
    if not isinstance(previous_status, dict) or previous_status.get('available') is not True:
        return dict(unavailable, reason='no_available_previous_intent',
                    unavailable_does_not_mean_no_past_attack=True)
    previous = previous_status.get('previous') or {}
    start, duration = previous.get('source_step'), previous.get('duration_frames')
    if not _integer(start) or not _integer(duration) or start < 0 or duration <= 0:
        return dict(unavailable, reason='invalid_previous_clock')
    if step < start or (scene_id is not None and str(previous.get('scene_id')) != str(scene_id)):
        return dict(unavailable, reason='previous_time_or_scene_mismatch')
    end = int(start + duration)
    age, after_end = int(step - start), int(step - end)
    event = previous.get('event_absolute_seconds')
    return dict(available=True, previous_source_step=int(start),
                previous_target_id=previous.get('target_id'), duration_frames=int(duration),
                nominal_expiry_step=end, current_source_step=int(step), dt_seconds=float(dt),
                age_since_start_frames=age, age_since_start_seconds=age * float(dt),
                nominal_lease_expired=after_end >= 0,
                age_since_expiration_seconds=after_end * float(dt) if after_end >= 0 else None,
                remaining_lease_seconds=max(-after_end, 0) * float(dt),
                previous_event_relative_to_now_seconds=float(event) - step * float(dt)
                if _finite(event) else None,
                accepted_intent_is_not_execution=True,
                expiry_is_nominal_not_observed_execution_end=True)


def source_facts(state, context, step, dt, scene_id=None):
    """只读当前快照；0.5米方位带与既有当前事实口径一致，缺失数据保持未知。"""
    if (not _integer(step) or step < 0 or context.get('source_step') != step
            or ('current_step' in state and state['current_step'] != step)):
        return dict(available=False, reason='source_step_mismatch', version=VERSION)
    if not _finite(dt) or dt <= 0:
        return dict(available=False, reason='invalid_dt', version=VERSION)
    ego = state.get('ego_state')
    if ego is None or len(ego) < 5 or not all(_finite(v) for v in ego[:5]):
        return dict(available=False, reason='invalid_current_ego', version=VERSION)
    rows, issues = {}, []
    for actor in state.get('agents', []):
        ident, vector = actor.get('id'), actor.get('state')
        if not _integer(ident) or str(ident) in rows:
            return dict(available=False, reason='ambiguous_actor_identity', version=VERSION)
        if vector is None or len(vector) < 4 or not all(_finite(v) for v in vector[:4]):
            issues.append(dict(target_id=int(ident), reason='missing_or_invalid_current_actor'))
            continue
        dx, dy = float(vector[0] - ego[0]), float(vector[1] - ego[1])
        yaw = float(ego[4]); distance = math.hypot(dx, dy)
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        right = dx * math.sin(yaw) - dy * math.cos(yaw)
        rate = (dx * float(vector[2] - ego[2]) + dy * float(vector[3] - ego[3])) / distance if distance else None
        rows[str(ident)] = dict(target_id=int(ident), forward_m=forward, right_m=right,
            longitudinal_label='ahead' if forward > .5 else 'behind' if forward < -.5 else 'aligned',
            lateral_label='right' if right > .5 else 'left' if right < -.5 else 'aligned',
            center_distance_m=distance, radial_gap_rate_mps=rate,
            current_distance_direction='undefined_same_center' if rate is None else
                'separating' if rate > 1e-6 else 'closing' if rate < -1e-6 else 'numerically_zero')
    prior = (context.get('rolling_intent') or {}).get('previous')
    return dict(available=True, version=VERSION, source_step=int(step), dt_seconds=float(dt),
        current_actor_positions=rows, position_label_deadband_m=.5,
        radial_numeric_tolerance_mps=1e-6, prior_intent_clock=lease_clock(prior, step, dt, scene_id),
        issues=issues, basis='current_measured_state_and_accepted_intent_metadata_only',
        future_or_executed_attack_not_certified=True, new_acceptance_gates=0)


class SourceFactContract:
    """包住最终请求，避免旧压缩提示链丢失事实；保持原 schema、预算和原始回答。"""
    def __init__(self, engine):
        self.engine = engine
        root = Path(__file__).resolve().parents[1] / 'skills/riskweaver'
        raw = (root / RULE).read_bytes()
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(raw).hexdigest()
        if manifest['files'].get(RULE) != digest:
            raise ValueError('source_fact_rule_hash_mismatch')
        self.document = dict(path=RULE, sha256=digest, text=raw.decode('utf-8'))
        self.facts = None

    def prepare_context(self, state, context, env):
        result = copy.deepcopy(context)
        self.facts = source_facts(state, result, int(env.current_step),
                                  float(env.dt), getattr(env, 'current_scene_id', None))
        result[FIELD] = copy.deepcopy(self.facts)
        return result

    def wrap(self, method):
        def controlled(planner, *args, **kwargs):
            request = planner._request_attack_plan
            receipts = []; self.facts = None
            def outgoing(prompt, scene_image=None):
                prefix, raw = prompt.rsplit('\n', 1)
                payload = json.loads(raw)
                if self.facts is None:
                    raise ValueError('source_facts_not_prepared_before_request')
                payload['context'][FIELD] = copy.deepcopy(self.facts)
                knowledge = payload['knowledge']
                # 消融的 none/protocol 层不额外获得推理规则；事实字段不改变目录顺序。
                if self.engine.registry.level in ('analysis', 'full'):
                    knowledge['documents'] = [d for d in knowledge['documents'] if d['path'] != RULE]
                    knowledge['documents'].append(copy.deepcopy(self.document))
                receipts.append(dict(source_step=self.facts.get('source_step'),
                    facts_sha256=_hash(self.facts), rule_sha256=self.document['sha256'],
                    rule_included=self.engine.registry.level in ('analysis', 'full')))
                return request(prefix + '\n' + json.dumps(payload, ensure_ascii=False,
                    allow_nan=False), scene_image=scene_image)
            planner._request_attack_plan = outgoing
            try:
                return method(planner, *args, **kwargs)
            finally:
                planner._request_attack_plan = request
                if isinstance(getattr(planner, 'last_trace', None), dict):
                    planner.last_trace['source_fact_contract'] = dict(version=VERSION,
                        actual_wire=receipts, response_schema_unchanged=True,
                        response_not_rewritten=True, candidate_order_unchanged=True,
                        extra_API_requests=0, new_acceptance_gates=0,
                        prose_correctness_not_certified=True)
        return controlled
