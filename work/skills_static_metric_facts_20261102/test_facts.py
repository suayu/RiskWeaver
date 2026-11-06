"""含真实114目录的排序反例；不请求模型、不生成新仿真。"""
import copy
import json
import unittest
from pathlib import Path

from facts import build_facts


def item(cid, area=3, **extra):
    return dict(obstacle_candidate_id=cid, geometric_free_area_m2=area,
        geometric_area_basis='same_grid', area_resolution_m=.5,
        background_CV_margin_m=extra.pop('margin', None), primitive_count=2, **extra)


class FactsTest(unittest.TestCase):
    def test_rank_and_ties_are_explicit_without_reordering(self):
        catalog = [item('a', 3), item('b', 1), item('c', 1), item('d', 2)]
        before = copy.deepcopy(catalog)
        facts = build_facts(catalog)
        rows = facts['groups'][0]['metrics']['geometric_free_area_m2']['candidates']
        self.assertEqual(rows['a']['competition_rank_ascending'], 4)
        self.assertEqual(rows['a']['dense_rank_ascending'], 3)
        self.assertEqual(rows['b']['equal_value_ids'], ['b', 'c'])
        self.assertEqual(rows['d']['smaller_ids'], ['b', 'c'])
        self.assertEqual(before, catalog)

    def test_null_and_nonfinite_values_are_not_zero_or_safe(self):
        facts = build_facts([item('a', None), item('b', float('nan'), margin=float('inf')), item('c', True)])
        for name in ['geometric_free_area_m2', 'background_CV_margin_m']:
            self.assertEqual(facts['groups'][0]['metrics'][name]['valid_count'], 0)
        json.dumps(facts, allow_nan=False)

    def test_incompatible_and_missing_scopes_are_separate(self):
        a, b, c, d = [item(str(i)) for i in range(4)]
        b['area_resolution_m'] = 1
        c['geometric_area_basis'] = None
        d['geometric_area_basis'] = None
        facts = build_facts([a, b, c, d])
        self.assertEqual(len(facts['groups']), 4)
        self.assertFalse(facts['groups'][2]['comparable_scope_known'])

    def test_partial_known_extrema_are_labelled(self):
        m = build_facts([item('a', 1), item('b', None)])['groups'][0]['metrics']['geometric_free_area_m2']
        self.assertFalse(m['all_values_known'])
        self.assertEqual(m['minimum_known_ids'], ['a'])

    def test_duplicate_ids_fail_loudly(self):
        with self.assertRaises(ValueError):
            build_facts([item('a'), item('a')])

    def test_empty_catalog_is_not_an_attack(self):
        self.assertEqual(build_facts([])['groups'], [])

    def test_real_scene114_third_not_second_or_minimum(self):
        path = Path('D:/RiskWeaver/outputs/skills_dynamic_static_20261101/experiments/skills_hybrid_rag_highlevel_review10_20261101_r1/results.json')
        if not path.exists():
            self.skipTest('历史原件未复制到此主机')
        row = next(r for r in json.loads(path.read_text(encoding='utf-8'))['results']
            if r['arm'] == 'skills_rag' and r['scene'] == 114)
        catalog = row['planner_trace']['context']['obstacle_candidates']
        metrics = build_facts(catalog)['groups'][0]['metrics']['geometric_free_area_m2']
        facts = metrics['candidates']['static_11']
        self.assertEqual(facts['value'], 581.5)
        self.assertEqual(facts['smaller_ids'], ['static_2', 'static_8'])
        self.assertEqual(facts['dense_rank_ascending'], 3)
        self.assertEqual(facts['competition_rank_ascending'], 3)
        self.assertNotIn('static_11', metrics['minimum_known_ids'])


if __name__ == '__main__':
    unittest.main()
