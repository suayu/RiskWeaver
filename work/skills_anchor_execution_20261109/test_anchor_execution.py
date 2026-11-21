"""在无模型环境检查完整时域、插值约束和工作器装配。"""

import os
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

import anchor_execution
from anchor_execution import (MAX_CURVATURE, check_dynamics, forward_headings, interpolate_joint,
                              interpolation_reference, pursue_forward, window_steps)
from run_anchor_worker import install


LIMITS = dict(max_speed=20., max_acceleration=6., max_jerk=20., max_step_distance=2.)
ANCHORS = [[0., 0.], [2., 0.], [4., 0.], [6., 0.]]


class Rejection(RuntimeError):
    pass


class AnchorTests(unittest.TestCase):
    def test_covers_all_anchors_and_tail(self):
        self.assertEqual(window_steps(.1), 30)
        points = interpolation_reference(ANCHORS, [2., 0.], [0., 1., 2., 3., 3.2])
        np.testing.assert_allclose(points, [[0, 0], [2, 0], [4, 0], [6, 0], [6.4, 0]])
        with self.assertRaisesRegex(ValueError, "off_simulation_grid"):
            window_steps(.3)

    def test_motion_boundary_jerk(self):
        positions = np.array([[.2, 0.], [.4, 0.]])
        state = np.array([0., 0., 2., 0., 0.])
        self.assertTrue(check_dynamics(positions, state, .1, LIMITS)[2])
        self.assertFalse(check_dynamics(positions, state, .1, LIMITS, [1.5, 0.])[2])

    def test_joint_preserves_target_and_background(self):
        snapshot = os.environ["RISKWEAVER_TEST_SOURCE"]
        sys.path.insert(0, snapshot)
        self.addCleanup(sys.path.remove, snapshot)
        from policies.traffic_types import JointTrajectory
        from policies.joint_safety import validate_execution_prefix

        gate_module = ModuleType("policies.ego_profile")
        gate_module.JointTrajectorySafetyGate = lambda cfg: NS(dynamic_limits=LIMITS)
        joint_safety = ModuleType("policies.joint_safety")
        joint_safety.NoSafeJointCandidate = Rejection
        joint_safety.validate_execution_prefix = validate_execution_prefix
        projection = ModuleType("policies.trajectory_projection")
        cfg = NS(prediction_horizon=32,
                 dynamics_projection=NS(max_deviation_m=.25, max_seconds_per_vehicle=2.))
        actors = np.array([[0, 0, 2, 0, 0, 4, 2], [0, 15, 3, 0, 0, 4, 2]], dtype=float)
        history = [actors]
        saved = []
        env = NS(cfg=NS(sim=NS(traffic_model=cfg)), data_dict={"agent": history},
                 agent_active=[True, True], dt=.1, current_step=0,
                 attack_intent=dict(target_id=0, anchors=ANCHORS, duration=30, source_step=0, request_id="r"),
                 get_scenario_frame=lambda: NS(static_obstacles_global=[]), inject_joint_trajectory=saved.append)
        with patch.dict(sys.modules, {"policies.ego_profile": gate_module,
                                     "policies.joint_safety": joint_safety,
                                     "policies.trajectory_projection": projection}):
            # 替身求解器分别返回可执行解和过度偏离解，验证提交前的独立复核。
            reference = interpolation_reference(ANCHORS, [2, 0], np.arange(1, 33) * .1)
            perturbed = reference.copy()
            perturbed[0, 0] += .2
            with patch.object(anchor_execution, "interpolation_reference", return_value=perturbed):
                projection.project_positions = lambda *args: (reference, None, None, {})
                self.assertTrue(interpolate_joint(env).metadata["interpolation"]["projected"])
                projection.project_positions = lambda *args: (perturbed, None, None, {})
                self.assertEqual(interpolate_joint(env).metadata["interpolation"]["follow_mode"], "forward_pursuit")
                with patch.object(anchor_execution, "pursue_forward", return_value=None):
                    with self.assertRaisesRegex(Rejection, "forward_pursuit_infeasible"):
                        interpolate_joint(env)
            for step in range(30):
                env.current_step = step
                joint = interpolate_joint(env)
                self.assertIsInstance(joint, JointTrajectory)
                self.assertEqual(joint.metadata["anchor_guidance_valid_steps"], 30 - step)
                next_states = history[-1].copy()
                next_states[:, :2], next_states[:, 2:4] = joint.positions_global[:, 0], joint.velocities_global[:, 0]
                history.append(next_states)
            np.testing.assert_allclose(history[-1][:, :2], [[6, 0], [9, 15]])
            env.current_step = 30
            with self.assertRaisesRegex(Rejection, "invalid_execution_window"):
                interpolate_joint(env)
            env.attack_intent = None
            self.assertFalse(interpolate_joint(env).metadata["anchor_guidance_active"])

    def test_reverse_anchor_uses_forward_maximum_turn(self):
        state = np.array([0., 0., 2., 0., 0.])
        reference = interpolation_reference([[0, 0], [-2, 0], [-4, 0], [-6, 0]],
                                             state[2:4], np.arange(1, 33)*.1)
        original_velocity = check_dynamics(reference, state, .1, LIMITS)[0]
        self.assertIsNone(forward_headings(original_velocity, state, .1))
        result = pursue_forward(reference, state, .1, LIMITS)
        self.assertIsNotNone(result)
        velocity, _, valid = check_dynamics(result, state, .1, LIMITS)
        self.assertTrue(valid)
        yaw = forward_headings(velocity, state, .1)
        self.assertIsNotNone(yaw)
        self.assertGreater(result[0, 0], 0.)
        self.assertGreater(yaw[0], 0.)
        self.assertAlmostEqual(yaw[0], MAX_CURVATURE * (.5*(2. + np.linalg.norm(velocity[0]))) * .1)
        self.assertTrue((np.abs(np.diff(np.r_[0., yaw])) < np.pi / 2).all())

    def test_worker_duration_and_guard(self):
        class Planner:
            def build_context(self, env):
                return {"candidates": [{"duration": 10}]}

        class Generator:
            last_attack_frame = 0

            def step(self, env, t):
                return "original_scheduler"

            def plan_and_inject(self, env):
                self.attack_duration = Planner().build_context(env)["candidates"][0]["duration"]
                env.attack_intent = dict(anchors=ANCHORS, duration=self.attack_duration, target_id=0,
                                         source_step=0)

        class Simulator:
            dt = .1
            current_step = 0
            agent_active = [True]
            attack_intent = None
            cfg = NS(sim=NS(traffic_model=NS(prediction_horizon=32)))

            def prepare_background_traffic(self):
                return NS(metadata={"anchor_guidance_active": True, "anchor_guidance_target_id": 0,
                                    "anchor_guidance_valid_steps": self.mask_steps,
                                    "replan_cache_offset": self.cache_offset})

        recipes = NS(proposals_for_actor=lambda ego, actor, target, dt, duration: duration)
        runner = NS(group_overrides=lambda: [])
        projection = NS(project_joint=lambda *a: None)
        sim, generator = Simulator(), Generator()
        original = Simulator.prepare_background_traffic
        with install("diffusion", runner, Planner, Generator, Simulator, recipes, projection, Rejection):
            self.assertEqual(recipes.proposals_for_actor(None, None, 0), 30)
            generator.plan_and_inject(sim)
            self.assertEqual(sim.attack_intent["anchor_times_s"], [0, 1, 2, 3])
            self.assertIs(generator.step(sim, 29), True)
            self.assertEqual(generator.step(sim, 30), "original_scheduler")
            sim.mask_steps, sim.cache_offset = 10, 0
            with self.assertRaisesRegex(Rejection, "remaining_anchor_window"):
                sim.prepare_background_traffic()
            sim.mask_steps, sim.cache_offset, sim.current_step = 30, 1, 1
            sim.prepare_background_traffic()
        self.assertIs(Simulator.prepare_background_traffic, original)
        self.assertEqual(Planner().build_context(sim)["candidates"][0]["duration"], 10)
        trace = NS(actual_guidance=lambda env: env.diffusion_controller.policy)
        original_guidance = trace.actual_guidance
        with install("interpolation", runner, Planner, Generator, Simulator, recipes, projection, Rejection,
                     trace_module=trace):
            self.assertIn("sim.traffic_model.backend=external_joint", runner.group_overrides())
            sim.traffic_backend, sim.diffusion_controller = "external_joint", None
            self.assertEqual(trace.actual_guidance(sim), dict(backend="kinematic_interpolation", params=None,
                                                             weights=None, sampling=None))
            sim.traffic_backend, sim.diffusion_controller = "safe_sim_diffusion", NS(policy="original")
            self.assertEqual(trace.actual_guidance(sim), "original")
            with self.assertRaisesRegex(ValueError, "no_diffusion_controller"):
                sim.prepare_background_traffic()
        self.assertIs(trace.actual_guidance, original_guidance)


if __name__ == "__main__":
    unittest.main()
