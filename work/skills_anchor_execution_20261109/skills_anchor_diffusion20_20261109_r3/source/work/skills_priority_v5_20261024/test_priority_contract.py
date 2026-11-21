"""用生产验证入口检查新响应契约与精确模型兼容，避免影响其他提供方。"""
import copy
import json
import unittest
from priority_planner import PrioritySkillsPlanner, compatible_request, render_priority_prompt
from evidence import candidate_evidence
from test_priority_evidence import STATE, CANDIDATE, response


class ContractTests(unittest.TestCase):
    def test_actual_validator_accepts_grounded_plan(self):
        engine = PrioritySkillsPlanner.__new__(PrioritySkillsPlanner)
        engine.priority_evidence = candidate_evidence(STATE, [CANDIDATE], .1, 1.)
        row = dict(response(), request_id='r', state_hash='s', catalog_hash='c',
                   scene_analysis='background target0', candidate_reasons={'x0':'active gap decreases'}, no_attack_reason='')
        engine.validate_response(row, 'r', 's', 'c', {'x0': CANDIDATE}, False)
        row['grounding']['x0']['target_role'] = 'ego'
        with self.assertRaises(ValueError):
            engine.validate_response(row, 'r', 's', 'c', {'x0': CANDIDATE}, False)

    def test_original_schema_is_not_silently_accepted_as_priority_proof(self):
        engine = PrioritySkillsPlanner.__new__(PrioritySkillsPlanner)
        with self.assertRaises(ValueError): engine.validate_response({}, 'r', 's', 'c', {}, False)

    def test_new_prefix_replaces_conflicting_old_exact_fields(self):
        p = render_priority_prompt('old exactly fields\n'+json.dumps({'context': {'candidates': []}}))
        self.assertNotIn('old exactly fields', p)
        self.assertIn('grounding', p)

    def test_exact_glm_model_is_adapted_without_mutating_request(self):
        original = dict(model='glm-5.3', messages=[], extra_body={'enable_thinking':False,'custom':1})
        result = compatible_request('dashscope', original)
        self.assertTrue(result['extra_body']['enable_thinking'])
        self.assertEqual(result['extra_body']['custom'], 1)
        self.assertFalse(original['extra_body']['enable_thinking'])

    def test_other_models_and_providers_keep_original_contract(self):
        for provider, model in [('deepseek','glm-5.3'), ('dashscope','kimi-k3'), ('dashscope','unknown-glm')]:
            kwargs = dict(model=model, extra_body={'enable_thinking':False})
            self.assertEqual(compatible_request(provider,kwargs),kwargs)


if __name__ == '__main__': unittest.main()
