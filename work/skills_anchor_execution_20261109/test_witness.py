"""验证规避见证的几何反例、空集放行口径和协议继承。"""

import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

from avoidance_witness import add_evidence, apply_verdict, search, swept_clear
from launch_anchor import ARM, ROOT, derive
from run_anchor_worker import install


def crossing_env():
    t = np.arange(1, 33) * .1
    joint = NS(source_step=0, agent_ids=np.array([0]),
               positions_global=np.array([np.column_stack((np.full(len(t), 8.), 8. - 3.*t))]),
               yaws_global=np.full((1, len(t), 1), -np.pi/2),
               valid_mask=np.ones((1, len(t)), dtype=bool))
    reach = NS(horizon_seconds=3., lane_half_width=4., max_speed=25.,
               longitudinal_accelerations=[-4., -2., 0., 2.])
    return NS(current_step=0, dt=.1, pending_joint_trajectory=joint, agent_active=[True],
              attack_intent={"target_id": 0}, get_static_obstacles=lambda: [],
              data_dict={"agent": [np.array([[8., 8., 0., -3., -np.pi/2, 2., 1.4]])]},
              ego_state=np.array([0., 0., 3., 0., 0., 2., 1.4]),
              cfg=NS(sim=NS(evaluation=NS(reachability=reach))),
              scenario_dict={"route": [[-10., 0.], [30., 0.]]})


class WitnessTests(unittest.TestCase):
    def test_wide_acceleration_finds_safe_witness(self):
        env = crossing_env()
        result = search(env, 4.)
        self.assertTrue(result["verified"], result)
        self.assertEqual(result["requested_accelerations_mps2"], [4., 4.])
        self.assertEqual(len(result["points_xy"]), 61)
        self.assertAlmostEqual(result["step_seconds"], .05)

    def test_swept_collision_between_endpoints(self):
        points = np.array([[-3., 0.], [3., 0.]])
        other = np.array([[[0., 0.], [0., 0.]]])
        self.assertFalse(swept_clear(points, np.zeros(2), [2., 1.], other,
                                     np.zeros((1, 2)), np.array([[2., 1.]])))

    def test_missing_prediction_and_initial_contact_fail_closed(self):
        env = crossing_env()
        env.pending_joint_trajectory.valid_mask[0, 5] = False
        self.assertEqual(search(env, 4.)["status"], "prediction_unavailable")
        env = crossing_env()
        env.get_static_obstacles = lambda: [dict(center=[0., 0.], yaw=0., length=2., width=2.)]
        self.assertFalse(search(env, 4.)["verified"])

    def test_score_and_gate_stay_separate(self):
        event = dict(reachability_backend="route_grid", source_step=0, original_area_m2=12.,
                     dangerous_area_m2=0., dangerous_solvable=False, difficulty=1.,
                     response_space_compression={"curve": [{"compression": 1., "remaining_area_ratio": 0.}]})
        verdict = dict(allowed=False, relative_measurement_valid=True, relative_difficulty=1.)
        add_evidence(event, crossing_env(), 4.)
        self.assertTrue(math.isnan(event["difficulty"]))
        self.assertIsNone(event["response_space_compression"]["curve"][0]["compression"])
        result = apply_verdict(verdict, event, "positive_area")
        self.assertTrue(result["allowed"])
        self.assertFalse(result["relative_measurement_valid"])
        self.assertIsNone(result["relative_difficulty"])
        self.assertFalse(apply_verdict(verdict, event, "legacy_relative")["allowed"])
        event["source_step"] = 1
        self.assertFalse(apply_verdict(verdict, event, "positive_area")["allowed"])

    def test_protocol_freezes_mode_without_changing_budget(self):
        folder = ROOT / "work/skills_acceptance_20261109"
        fixture = folder / "protocol_r2/protocol.json"
        if not fixture.is_file():
            fixture = folder / "frozen_r2_protocol.json"
        base = json.loads(fixture.read_text(encoding="utf-8-sig"))
        command = base["commands"][ARM]
        args = derive(base, command, "cpu_interpolation", Path("protocol"), Path("experiment"), 4.)
        built = args["commands"][ARM]
        self.assertEqual(built[built.index("--device")+1], "cpu")
        self.assertIn("--interpolation-cpu", built)
        self.assertEqual(built[built.index("--witness-max-accel-mps2")+1], "4.0")
        self.assertEqual(built[built.index("--transport-limit")+1], "4")
        self.assertEqual(args["planned_scenes"], base["planned_scenes"])
        self.assertEqual(command[command.index("--device")+1], "cuda")
        self.assertEqual(args["evaluation_scope"], "skills_interpolation_acceptance")

    def test_worker_routes_witness_to_gate_and_restores_legacy(self):
        class Generator:
            step = plan_and_inject = lambda *args: None

            def evaluate_reachability(self, env):
                return dict(reachability_backend="route_grid", source_step=env.current_step,
                            dangerous_area_m2=0., dangerous_solvable=False, difficulty=1.)

        planner = NS(build_context=lambda: None)
        simulator = NS(prepare_background_traffic=lambda: None)
        recipes = NS(proposals_for_actor=lambda: None)
        projection = NS(project_joint=lambda: None)
        base = lambda event, policy: dict(allowed=False)
        runner = NS(assess_reachability_execution=base, group_overrides=lambda: [])
        env = crossing_env()
        env.cfg.sim.evaluation.reachability.execution_gate_policy = "positive_area"
        with install("diffusion", runner, planner, Generator, simulator, recipes, projection,
                     RuntimeError, witness_accel=4.):
            event = Generator().evaluate_reachability(env)
            self.assertTrue(runner.assess_reachability_execution(event, "positive_area")["allowed"])
        self.assertIs(runner.assess_reachability_execution, base)
        self.assertNotIn("avoidance_witness", Generator().evaluate_reachability(env))


if __name__ == "__main__":
    unittest.main()
