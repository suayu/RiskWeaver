"""验证身份/有效期/代理证据边界，避免把远期锚点当作已执行威胁。"""
import copy
import math
import unittest
from evidence import candidate_evidence, validate_grounding


STATE = dict(ego_state=[0, 0, 0, 0, 0], agents=[dict(id=0, state=[20, 0, 0, 0, 0])])
CANDIDATE = dict(candidate_id='x0', target_id=0, anchors=[[20, 0], [18, 0], [10, 0], [0, 0]], duration=10)


def response():
    return dict(ranking=['x0'], grounding={'x0': dict(target_id=0, target_role='background_vehicle',
        active_horizon_seconds=1., evidence_ids=['x0:actor','x0:validity','x0:active'], interaction_basis='active_window_proxy')})


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.evidence = candidate_evidence(STATE, [CANDIDATE], .1, 1.)

    def test_background_zero_is_not_ego(self):
        self.assertEqual(self.evidence['x0']['facts']['x0:actor']['role'], 'background_vehicle')
        row = response(); row['grounding']['x0']['target_role'] = 'ego'
        with self.assertRaises(ValueError): validate_grounding(row, {'x0': CANDIDATE}, self.evidence)

    def test_active_window_cannot_claim_terminal_three_second_position(self):
        facts = self.evidence['x0']['facts']
        self.assertEqual(facts['x0:active']['min_center_distance_m'], 18.)
        self.assertEqual(facts['x0:after']['min_center_distance_m'], 0.)
        self.assertFalse(facts['x0:active']['executed'])
        self.assertFalse(facts['x0:active']['collision_or_evasion_guaranteed'])

    def test_grounding_rejects_wrong_duration_unknown_id_and_missing_fact(self):
        for key, val in [('active_horizon_seconds', 3.), ('evidence_ids', ['x0:actor','x0:validity','fake']), ('evidence_ids', ['x0:active'])]:
            row = response(); row['grounding']['x0'][key] = val
            with self.subTest(key=key, val=val), self.assertRaises(ValueError):
                validate_grounding(row, {'x0': CANDIDATE}, self.evidence)

    def test_later_forecast_requires_later_reference(self):
        row = response(); row['grounding']['x0']['interaction_basis'] = 'later_forecast_only'
        with self.assertRaises(ValueError): validate_grounding(row, {'x0': CANDIDATE}, self.evidence)
        row['grounding']['x0']['evidence_ids'][-1] = 'x0:after'
        validate_grounding(row, {'x0': CANDIDATE}, self.evidence)

    def test_time_aligned_cv_prevents_static_ego_shortcut(self):
        state = copy.deepcopy(STATE); state['ego_state'][2] = 10
        facts = candidate_evidence(state, [CANDIDATE], .1, 1.)['x0']['facts']
        self.assertAlmostEqual(facts['x0:active']['end_center_distance_m'], 8.)

    def test_whole_scene_rigid_transform_keeps_evidence(self):
        state = copy.deepcopy(STATE); c = copy.deepcopy(CANDIDATE)
        for point in c['anchors']:point[:] = [-point[1]+3, point[0]-7]
        state['ego_state'][:2] = [3, -7]; state['agents'][0]['state'][:2] = [3, 13]
        new = candidate_evidence(state, [c], .1, 1.)
        self.assertAlmostEqual(new['x0']['facts']['x0:active']['min_center_distance_m'], 18.)

    def test_bad_actor_or_nonfinite_timebase_is_not_repaired(self):
        c = copy.deepcopy(CANDIDATE); c['target_id'] = 123
        with self.assertRaises(ValueError): candidate_evidence(STATE, [c], .1, 1.)
        with self.assertRaises(ValueError): candidate_evidence(STATE, [CANDIDATE], math.nan, 1.)


if __name__ == '__main__': unittest.main()
