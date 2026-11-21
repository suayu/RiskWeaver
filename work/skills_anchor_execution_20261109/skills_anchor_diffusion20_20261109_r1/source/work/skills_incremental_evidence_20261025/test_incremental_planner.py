"""真实响应校验入口的方向负例，以及不向模型重复传完整失败目录。"""
import copy
import json
from types import SimpleNamespace as NS
import unittest
import numpy as np
from omegaconf import OmegaConf
from incremental_planner import IncrementalSkillsPlanner, render_incremental_prompt, ROOT
from policies.llm_adversarial_planner import LLMAdversarialPlanner


def fixture():
    planner = object.__new__(IncrementalSkillsPlanner)
    planner.incremental_cards = {'x0':dict(proxy_direction='farther_than_natural_CV')}
    facts = {'x0:actor':{}, 'x0:validity':{}, 'x0:active':{}, 'x0:contrast':{},
             'x0:safety':dict(supplied_margin_m=None, semantic_metrics={'unnecessary_repeat':9})}
    planner.priority_evidence = {'x0':dict(target_id=0, active_horizon_seconds=1., facts=facts)}
    catalog = {'x0':dict(target_id=0)}
    response = dict(request_id='r', state_hash='s', catalog_hash='c', ranking=['x0'],
        scene_analysis='Separate ego and background target0.', candidate_reasons={'x0':'Exploratory only.'},
        no_attack_reason='', grounding={'x0':dict(target_id=0, target_role='background_vehicle',
        active_horizon_seconds=1., evidence_ids=['x0:actor','x0:validity','x0:active','x0:contrast'],
        interaction_basis='exploratory_probe', contrast_direction='farther_than_natural_CV')})
    return planner, catalog, response


class IncrementalPlannerTests(unittest.TestCase):
    def test_actual_validator_rejects_false_effect_direction(self):
        planner, catalog, response = fixture()
        planner.validate_response(response, 'r', 's', 'c', catalog, False)
        wrong = copy.deepcopy(response);wrong['grounding']['x0']['contrast_direction'] = 'closer_than_natural_CV'
        with self.assertRaisesRegex(ValueError, 'contradicts_fact'):
            planner.validate_response(wrong, 'r', 's', 'c', catalog, False)
        self.assertIn('contrast_direction', response['grounding']['x0'])

    def test_extra_or_missing_reference_not_accepted(self):
        planner, catalog, response = fixture();response['grounding']['x0']['evidence_ids'].remove('x0:contrast')
        with self.assertRaisesRegex(ValueError, 'reference_missing'):
            planner.validate_response(response, 'r', 's', 'c', catalog, False)

    def test_display_omits_audit_without_modifying_original(self):
        planner, _, _ = fixture()
        card = dict(candidate_min_gap_m=9., candidate_min_time_s=1., natural_CV_min_gap_m=8.,
             natural_CV_min_time_s=1., min_gap_difference_m=-1., same_time_difference_at_candidate_min_m=-1.,
             proxy_direction='farther_than_natural_CV')
        payload = dict(context=dict(priority_evidence=planner.priority_evidence, incremental_cards={'x0':card},
                         incremental_recipe_audit=[{'raw_rejected_geometry':list(range(100))}]))
        before = copy.deepcopy(payload)
        text = render_incremental_prompt('unused prefix\n'+json.dumps(payload))
        decoded = json.loads(text.split('\n')[-1])
        self.assertNotIn('incremental_recipe_audit', decoded['context'])
        self.assertNotIn('incremental_cards', decoded['context'])
        self.assertNotIn('semantic_metrics', decoded['context']['priority_evidence']['x0']['facts']['x0:safety'])
        self.assertIn('interaction_basis, contrast_direction}', text)
        self.assertEqual(payload, before)

    def test_multiline_upstream_prefix_and_full_plan_boundary(self):
        # 真实的Compact→Priority→Incremental调用链；传输替身不初始化SDK或读取凭据。
        prior = ROOT/'experiments/skills_priority_v5_frozen_development10_pair_20261024_r1'
        item = next(x for x in json.loads((prior/'inputs.json').read_text()) if x['scenario_index'] == 74)
        state, frame = item['request']['state'], item['frame']
        cfg = OmegaConf.load(ROOT/'experiments/skills_modelpool_recovery_development10_paired_20261023_r3/skills_full_shared_recovery/resolved_config.yaml')
        route = np.asarray(state['route'])
        env = NS(ego_state=np.asarray(frame['ego']), data_dict={'agent':[np.asarray(frame['agents'])]},
            agent_active=np.asarray(frame['active']), current_step=0, current_scene_id='scenario_074',
            scenario_dict={'route':route, 'lanes':[route]}, get_static_obstacles=lambda:[], cfg=cfg,
            dt=.1, pending_joint_trajectory=None, get_state_for_planning=lambda:state)
        for extra in [False, True]:
            engine = IncrementalSkillsPlanner(cfg.sim, 4, extra_recipes=extra)
            planner = object.__new__(LLMAdversarialPlanner)
            planner.allow_trajectory = True;planner.use_obstacles = False
            planner.last_request_failed = False;planner.last_trace = {}
            seen = []
            def request(prompt, scene_image=None):
                payload = json.loads(prompt.rsplit('\n', 1)[1]);seen.append(payload)
                candidate = payload['context']['candidates'][0];cid = candidate['candidate_id']
                evidence = payload['context']['priority_evidence'][cid]
                response = dict(request_id=payload['request_id'], state_hash=payload['state_hash'], catalog_hash=payload['catalog_hash'],
                    ranking=[cid], scene_analysis='Fixture only, no claim of actual threat.', candidate_reasons={cid:'Exploratory proxy.'}, no_attack_reason='',
                    grounding={cid:dict(target_id=candidate['target_id'], target_role='background_vehicle', active_horizon_seconds=1.,
                    evidence_ids=[cid+':actor',cid+':validity',cid+':active',cid+':contrast'], interaction_basis='exploratory_probe',
                    contrast_direction=evidence['facts'][cid+':contrast']['proxy_direction'])})
                return '', response
            planner._request_attack_plan = request
            plan = engine.plan(planner, env, state, ['pure dynamic'])
            self.assertTrue(plan and plan['attack'], planner.last_trace)
            self.assertEqual(len(seen), 1)
            self.assertEqual(planner.last_trace['incremental_protocol']['direction_validation'], 'accepted')
            self.assertNotIn('incremental_recipe_audit', seen[0]['context'])
            self.assertIs(planner._request_attack_plan, request)


if __name__ == '__main__':unittest.main()
