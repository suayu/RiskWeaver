"""从旧 D 的基础测量构造规避压力，场景结果单独计分。"""
import math

import numpy as np

from experiments.comparison.e1_metrics import obb_ttc
from scoring_observer import FormalRiskObserver, has_positive_reachable_area
from policies.risk_metrics import AdversarialRiskMetrics, compute_scenario_danger_score
from dual_track_evaluation import VERSION


def pressure_score(ttc, ea, reachability, composite=None):
    """沿用旧 D 逐帧归一化，只传入已确认的规避压力分量。"""
    if (ttc is None or math.isnan(ttc) or ttc < 0 or ea is None
            or not math.isfinite(ea) or reachability is None
            or not math.isfinite(reachability.get("difficulty", math.nan))
            or reachability.get("dangerous_solvable") is None):
        return None
    compression = reachability["difficulty"]
    return compute_scenario_danger_score({
        "avg_min_ttc": ttc,
        "max_evasive_acceleration_mps2": ea,
        "mean_evasive_acceleration_mps2": ea,
        "reachability_event_count": 1,
        "reachability_difficulty": compression,
        "mean_reachability_difficulty": compression,
        "dangerous_scene_solvable": float(reachability["dangerous_solvable"]),
    }, composite)


def source_obb_ttc(env):
    """对同一源帧的活动车辆和已生效静态物体检查首次接触。"""
    ego = np.asarray(env.ego_state, dtype=float)
    states = np.asarray(env.data_dict["agent"][-1], dtype=float)
    active = np.asarray(env.agent_active, dtype=bool)
    values = [obb_ttc(ego, state) for state in states[active]]
    for obstacle in env.get_static_obstacles():
        x, y = obstacle["center"]
        values.append(obb_ttc(ego, [x, y, 0, 0, obstacle["yaw"],
                                    obstacle["length"], obstacle["width"]]))
    return min(values, default=math.inf)


class DualObserver:
    """每个实际执行的转移登记同一源帧的规避压力。"""

    latest = None
    danger_ttc_input_basis = 'post_transition_obb_ttc_dynamic_and_static'

    def __init__(self, evaluation_cfg):
        self.risk = AdversarialRiskMetrics(evaluation_cfg)
        # 只为独立评分器启用 EA，生成器沿用此前的运行配置。
        self.risk.ea_enabled = True
        # 旧 D 用干预前 EA，I 用干预后 EA，分别保留航向历史。
        self.legacy = FormalRiskObserver(evaluation_cfg)
        self.legacy.risk.ea_enabled = True
        self.composite = getattr(evaluation_cfg, "composite", None)
        self.reset()
        DualObserver.latest = self

    def reset(self):
        self.legacy.reset()
        self.records = self.legacy.records
        self.risk.reset_episode()
        self.initial_static_ids = None

    def begin_frame(self, env, expect_joint_prediction=True):
        self.env = env
        if self.initial_static_ids is None:
            self.initial_static_ids = {obj["id"] for obj in env.get_static_obstacles()}
        self.legacy.begin_frame(env, expect_joint_prediction)

    def finish_prediction(self, env, forecast=None, source="native", error=None):
        # 共用旧观察器的同源帧可达性，避免重复运行网格求解。
        reachability = self.legacy.finish_prediction(env, forecast, source, error)
        if self.attack_requires_rejection(reachability):
            # 动态意图和新增静态障碍物都必须在实际状态转移前通过门控。
            from policies.joint_safety import NoSafeJointCandidate
            raise NoSafeJointCandidate("riskweaver_attack_nonpositive_or_unverified_reachable_area")
        ea = self.risk.evaluate_ea(env)
        try:
            ttc = source_obb_ttc(env)
        except (ValueError, IndexError, KeyError, TypeError):
            ttc = None
        score = pressure_score(ttc, ea, reachability, self.composite)
        self.pending_challenge = dict(
            score=score, valid=score is not None,
            ea_mps2=ea if ea is not None and math.isfinite(ea) else None,
            source_obb_ttc_s=ttc if ttc is not None and math.isfinite(ttc) else None,
            ttc_no_contact_in_horizon=ttc == math.inf,
        )
        return reachability

    def attack_requires_rejection(self, reachability):
        # 尊重用户当前暂停面积门控的配置；评分无效不等于拒绝执行。
        sim = getattr(getattr(self.env, 'cfg', None), 'sim', None)
        if sim is not None and not bool(getattr(sim, 'reachable_area_gate_enabled', True)):
            return False
        # 原场景自带的静态物体不当作本次攻击。
        attack_active = bool(getattr(self.env, "attack_intent", None)) or any(
            obj["id"] not in self.initial_static_ids for obj in self.env.get_static_obstacles()
        )
        return attack_active and not has_positive_reachable_area(reachability)

    def finish_transition(self, info, min_ttc):
        # 通用运行器的第二参数仍为历史 TCA；新协议独立读取执行后 OBB TTC。
        try:
            ttc = source_obb_ttc(self.env)
        except (ValueError, IndexError, KeyError, TypeError):
            ttc = None
        record = self.legacy.finish_transition(info, ttc, feasible_only=True)
        record["outcome_ttc_basis"] = self.danger_ttc_input_basis
        record["outcome_obb_ttc_s"] = ttc if ttc is not None and math.isfinite(ttc) else None
        record["challenge"] = self.pending_challenge
        return record

    def result(self):
        return self.legacy.result()

    def challenge_result(self):
        valid = [r["challenge"]["score"] for r in self.records if r["challenge"]["valid"]]
        total = len(self.records)
        coverage = len(valid) / total if total else 0.
        passed = bool(valid)
        return dict(valid=passed, score=float(np.mean(valid)) if passed else None,
                    valid_frames=len(valid), total_frames=total, coverage=coverage,
                    minimum_coverage=None, version=VERSION + "_stationary_v1",
                    scope="conditional_source_states_legacy_pressure_components",
                    collision_outcome_used=False, off_route_outcome_used=False,
                    legacy_scoring_computed=False)
