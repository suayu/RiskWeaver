"""围绕自然接近误归因、不同最小时刻及原筛查拒绝的独立回归。"""
import copy
from types import SimpleNamespace as NS
import unittest
import numpy as np
from incremental_evidence import incremental_card, factual_summary
from active_window_recipes import proposals_for_actor, screened_proposals


def example():
    return dict(ego_state=[0, 0, 0, 0, 0], agents=[dict(id=0, state=[10, 0, -2, 0, 3.14])]), dict(
        candidate_id='x0', target_id=0, duration=10, anchors=[[10, 0], [8, 0], [6, 0], [4, 0]])


class IncrementalTests(unittest.TestCase):
    def test_natural_closing_is_not_attack_benefit(self):
        state, candidate = example()
        card = incremental_card(state, candidate)
        self.assertEqual(card['candidate_min_gap_m'], 8.)
        self.assertEqual(card['min_gap_difference_m'], 0.)
        self.assertFalse(card['actual_attack_benefit_measured'])

    def test_braking_oncoming_can_reduce_threat_proxy(self):
        state, candidate = example();candidate['anchors'] = [[10, 0], [9, 0], [9, 0], [10, 0]]
        self.assertEqual(incremental_card(state, candidate)['min_gap_difference_m'], -1.)

    def test_after_expiry_approach_cannot_improve_active_proxy(self):
        state, candidate = example();candidate['anchors'][2:] = [[0, 0], [0, 0]]
        self.assertEqual(incremental_card(state, candidate)['min_gap_difference_m'], 0.)

    def test_different_minimum_times_are_explicit(self):
        state, candidate = example();state['agents'][0]['state'] = [1, 0, -2, 0, 3.14]
        candidate['anchors'] = [[1, 0], [-.25, 0], [0, 0], [0, 0]]
        card = incremental_card(state, candidate)
        self.assertNotEqual(card['natural_CV_min_time_s'], card['candidate_min_time_s'])
        self.assertNotEqual(card['min_gap_difference_m'], card['same_time_difference_at_candidate_min_m'])

    def test_rigid_transform_preserves_card_and_input(self):
        state, candidate = example();before = copy.deepcopy((state, candidate));old = incremental_card(state, candidate)
        self.assertEqual((state, candidate), before)
        for s in [state['ego_state'], state['agents'][0]['state']]:
            s[:2] = [-s[1]+3, s[0]-7];s[2:4] = [-s[3], s[2]]
        candidate['anchors'] = [[-y+3, x-7] for x, y in candidate['anchors']]
        new = incremental_card(state, candidate)
        self.assertAlmostEqual(old['min_gap_difference_m'], new['min_gap_difference_m'])
        self.assertIn('背景车0', factual_summary(new))

    def test_missing_identity_invalid_source_and_partial_horizon(self):
        state, candidate = example();candidate['target_id'] = True
        with self.assertRaises(ValueError):incremental_card(state, candidate)
        candidate['target_id'] = 0;candidate['anchors'][0] = [9, 0]
        with self.assertRaises(ValueError):incremental_card(state, candidate)
        state, candidate = example();candidate['duration'] = 40
        card = incremental_card(state, candidate)
        self.assertFalse(card['full_active_window_covered']);self.assertEqual(card['observed_proxy_seconds'], 3.)


class RecipeTests(unittest.TestCase):
    def test_oncoming_advance_present_and_sparse_caps_preserved(self):
        proposals = proposals_for_actor([0, 0, 10, 0, 0], [30, 0, -8, 0, np.pi], 0)
        self.assertTrue(proposals)
        self.assertTrue(any(p['strategy'] == 'sudden_acceleration' for p in proposals))
        for p in proposals:
            anchors = np.asarray(p['anchors']);self.assertEqual(anchors.shape, (4, 2))
            np.testing.assert_array_equal(anchors[0], [30, 0])
            self.assertLessEqual(np.linalg.norm(np.diff(anchors, n=2, axis=0), axis=1).max(), 3.8+1e-8)
            self.assertFalse(p['execution_approved'])
            self.assertEqual(p['motion_relation'], 'opposing_headings')

    def test_screen_and_semantic_rejections_are_not_bypassed(self):
        env = NS(ego_state=np.array([0, 0, 0, 0, 0]), dt=.1, agent_active=[True],
                 data_dict={'agent':[np.array([[10, 0, -2, 0, np.pi]])]}, scenario_dict={'lanes':[]})
        state, _ = example()
        builder = NS(screen=lambda *args:(None, 'background_margin'))
        candidates, audit = screened_proposals(env, state, builder, lambda *args:self.fail('semantic called after screen rejection'))
        self.assertEqual(candidates, []);self.assertTrue(all(a['screen_rejection'] == 'background_margin' for a in audit))
        builder.screen = lambda *args:({}, None)
        candidates, audit = screened_proposals(env, state, builder, lambda *args:dict(accepted=False, reason='semantic_failure', metrics={}))
        self.assertEqual(candidates, []);self.assertTrue(all(a['semantic_validation']['reason'] == 'semantic_failure' for a in audit))

    def test_recipe_invalid_limits_and_stationary_target(self):
        with self.assertRaises(ValueError):proposals_for_actor([0]*5, [0]*5, 0, cap_values=(4.,))
        with self.assertRaises(ValueError):proposals_for_actor([0]*5, [0]*5, 0, duration=40)
        self.assertTrue(proposals_for_actor([0, 0, 0, 0, 0], [10, 0, 0, 0, np.pi], 0))


if __name__ == '__main__':unittest.main()
