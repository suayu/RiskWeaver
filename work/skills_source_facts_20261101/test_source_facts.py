"""核对真实帧时钟、坐标约定和最终请求补充；不调用 API 或模拟器。"""
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('source_fact_subject', HERE/'source_after/policies/skills_source_facts.py')
subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)


def previous(start=55, duration=10, scene='scene'):
    return dict(available=True, elapsed_seconds=1.1, previous_lease_expired=True,
        previous=dict(source_step=start, duration_frames=duration, scene_id=scene,
                      target_id=1, event_absolute_seconds=7.5))


class SourceFactTests(unittest.TestCase):
    def test_age_is_not_expiration_age(self):
        clock = subject.lease_clock(previous(), 66, .1, 'scene')
        self.assertEqual(clock['age_since_start_seconds'], 1.1)
        self.assertEqual(clock['age_since_expiration_seconds'], .1)
        self.assertEqual(clock['nominal_expiry_step'], 65)
        self.assertEqual(clock['remaining_lease_seconds'], 0)

    def test_expiration_boundary(self):
        clock = subject.lease_clock(previous(), 65, .1, 'scene')
        self.assertTrue(clock['nominal_lease_expired'])
        self.assertEqual(clock['age_since_expiration_seconds'], 0)

    def test_unexpired_does_not_have_expiration_age(self):
        clock = subject.lease_clock(previous(), 64, .1, 'scene')
        self.assertFalse(clock['nominal_lease_expired'])
        self.assertIsNone(clock['age_since_expiration_seconds'])
        self.assertEqual(clock['remaining_lease_seconds'], .1)

    def test_different_dt_uses_actual_step(self):
        clock = subject.lease_clock(previous(), 66, .2, 'scene')
        self.assertEqual(clock['age_since_start_seconds'], 2.2)
        self.assertEqual(clock['age_since_expiration_seconds'], .2)

    def test_unavailable_is_unknown(self):
        for status in (None, {}, dict(available=False, reason='memory_expired')):
            clock = subject.lease_clock(status, 66, .1, 'scene')
            self.assertFalse(clock['available'])
            self.assertNotIn('age_since_expiration_seconds', clock)
            self.assertTrue(clock['unavailable_does_not_mean_no_past_attack'])

    def test_stale_and_malformed_clocks(self):
        for status, step, dt, scene in ((previous(), 54, .1, 'scene'),
                (previous(), 66, .1, 'other'), (previous(duration=0), 66, .1, 'scene'),
                (previous(start=True), 66, .1, 'scene'), (previous(), 66, float('nan'), 'scene')):
            self.assertFalse(subject.lease_clock(status, step, dt, scene)['available'])

    def test_expired_event_can_be_negative(self):
        status = previous(); status['previous']['event_absolute_seconds'] = 6
        self.assertAlmostEqual(subject.lease_clock(status, 66, .1, 'scene')['previous_event_relative_to_now_seconds'], -.6)

    def facts(self, actor=None, step=66, **context):
        state = dict(ego_state=[0, 0, 2, 0, 0], current_step=step,
                     agents=[dict(id=1, state=actor or [-2, -.171, -1, 0, 0])])
        ctx = dict(source_step=step, rolling_intent=dict(previous=previous()), **context)
        return subject.source_facts(state, ctx, step, .1, 'scene')

    def test_deadband_prevents_false_left_or_right(self):
        fact = self.facts()['current_actor_positions']['1']
        self.assertEqual(fact['longitudinal_label'], 'behind')
        self.assertEqual(fact['lateral_label'], 'aligned')
        self.assertGreater(fact['right_m'], 0)

    def test_position_and_radial_motion_are_independent(self):
        closing = self.facts(actor=[-2, -2, 4, 0, 0])['current_actor_positions']['1']
        self.assertEqual(closing['lateral_label'], 'right')
        self.assertEqual(closing['current_distance_direction'], 'closing')
        self.assertEqual(self.facts()['current_actor_positions']['1']['current_distance_direction'], 'separating')

    def test_same_center_is_not_stationary_certificate(self):
        row = self.facts(actor=[0, 0, 0, 0, 0])['current_actor_positions']['1']
        self.assertIsNone(row['radial_gap_rate_mps'])
        self.assertEqual(row['current_distance_direction'], 'undefined_same_center')

    def test_rotation_translation_preserves_labels_and_distances(self):
        original = dict(ego_state=[1, 2, 2, 3, .4], agents=[dict(id=1, state=[-2, 4, 5, 1, 0])])
        expected = subject.source_facts(original, dict(source_step=0), 0, .1)['current_actor_positions']['1']
        for angle in (-2, -.5, .3, 1.9):
            c, s = math.cos(angle), math.sin(angle)
            def transform(v):
                x, y, vx, vy, yaw = v
                return [c*x-s*y+123, s*x+c*y-57, c*vx-s*vy, s*vx+c*vy, yaw+angle]
            moved = dict(ego_state=transform(original['ego_state']),
                         agents=[dict(id=1, state=transform(original['agents'][0]['state']))])
            actual = subject.source_facts(moved, dict(source_step=0), 0, .1)['current_actor_positions']['1']
            for field in ('right_m', 'forward_m', 'radial_gap_rate_mps', 'center_distance_m'):
                self.assertAlmostEqual(expected[field], actual[field])
            for field in ('longitudinal_label', 'lateral_label', 'current_distance_direction'):
                self.assertEqual(expected[field], actual[field])

    def test_inputs_untouched_and_missing_not_zero(self):
        state = dict(ego_state=[0, 0, 0, 0, 0], agents=[dict(id=1, state=None)])
        context = dict(source_step=0); old = copy.deepcopy((state, context))
        result = subject.source_facts(state, context, 0, .1)
        self.assertEqual((state, context), old)
        self.assertEqual(result['current_actor_positions'], {})
        self.assertEqual(len(result['issues']), 1)

    def test_snapshot_identity_and_nonfinite_rejected_as_unknown(self):
        self.assertFalse(subject.source_facts(dict(ego_state=[0]*5), dict(source_step=2), 1, .1)['available'])
        self.assertFalse(subject.source_facts(dict(ego_state=[float('inf')]*5), dict(source_step=0), 0, .1)['available'])

    def test_actual_request_survives_compression_and_does_not_rewrite_response(self):
        engine = NS(registry=NS(level='full')); bridge = subject.SourceFactContract(engine)
        env = NS(current_step=66, dt=.1, current_scene_id='scene')
        state = dict(ego_state=[0]*5, agents=[])
        context = dict(source_step=66, rolling_intent=dict(previous=previous()), candidates=[dict(candidate_id='keep')])
        response = object(); captured = []
        def request(prompt, scene_image=None):
            captured.append(json.loads(prompt.rsplit('\n', 1)[1])); return response
        planner = NS(_request_attack_plan=request, last_trace={})
        def inner(planner):
            prepared = bridge.prepare_context(state, context, env)
            self.assertIn(subject.FIELD, prepared)
            compressed = dict(context=dict(candidates=prepared['candidates']), knowledge=dict(documents=[]))
            return planner._request_attack_plan('prefix\n'+json.dumps(compressed))
        self.assertIs(bridge.wrap(inner)(planner), response)
        self.assertIs(planner._request_attack_plan, request)
        sent = captured[0]
        self.assertEqual(sent['context']['candidates'], context['candidates'])
        self.assertEqual(sent['context'][subject.FIELD]['prior_intent_clock']['age_since_expiration_seconds'], .1)
        self.assertEqual(sent['knowledge']['documents'][0]['path'], subject.RULE)
        self.assertEqual(len(planner.last_trace['source_fact_contract']['actual_wire']), 1)

    def test_wrapper_restored_after_failure(self):
        bridge = subject.SourceFactContract(NS(registry=NS(level='full')))
        def request(*args, **kwargs):raise RuntimeError('SDK failure')
        planner = NS(_request_attack_plan=request, last_trace={})
        def inner(planner):
            bridge.prepare_context(dict(ego_state=[0]*5), dict(source_step=0), NS(current_step=0, dt=.1))
            return planner._request_attack_plan('x\n'+json.dumps(dict(context={}, knowledge=dict(documents=[]))))
        with self.assertRaisesRegex(RuntimeError, 'SDK failure'):bridge.wrap(inner)(planner)
        self.assertIs(planner._request_attack_plan, request)

    def test_ablation_does_not_inject_rule(self):
        bridge = subject.SourceFactContract(NS(registry=NS(level='none')))
        captured = []; planner = NS(last_trace={}, _request_attack_plan=lambda prompt, **k: captured.append(json.loads(prompt.rsplit('\n', 1)[1])))
        def inner(planner):
            bridge.prepare_context(dict(ego_state=[0]*5), dict(source_step=0), NS(current_step=0, dt=.1))
            return planner._request_attack_plan('x\n'+json.dumps(dict(context={}, knowledge=dict(documents=[]))))
        bridge.wrap(inner)(planner)
        self.assertEqual(captured[0]['knowledge']['documents'], [])
        self.assertIn(subject.FIELD, captured[0]['context'])


if __name__ == '__main__':
    unittest.main()
