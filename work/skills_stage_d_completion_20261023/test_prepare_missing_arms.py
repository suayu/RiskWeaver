"""验证消融契约，防止候选扩展、隐藏重复参数或预算差异污染比较。"""
import unittest
from pathlib import Path
from prepare_missing_arms import COMMON, missing_command, value


def template():
    command = ['python', 'run_recovery_v4.py']
    for k, v in COMMON.items():
        command += [k, v]
    for k in ('--skills', '--compact-view', '--event-budget', '--reachability-preflight', '--route-tail-preflight'):
        command += [k, 'on']
    command += ['--max-screening-queries', '40', '--skill-level', 'full', '--candidate-expansion', 'on',
                '--manifest', 'old_manifest', '--output-dir', 'old_output']
    return command


class ProtocolTests(unittest.TestCase):
    def test_knowledge_ablation_changes_only_documents_and_output(self):
        old = missing_command(template(), 'unified_catalog_no_knowledge', Path('m'), Path('a'))
        new = missing_command(template(), 'skills_no_expansion', Path('m'), Path('a'))
        changes = [(a, b) for a, b in zip(old, new) if a != b]
        self.assertEqual(changes, [('none', 'full')])
        self.assertEqual(value(old, '--candidate-expansion'), 'off')
        self.assertEqual(value(new, '--candidate-expansion'), 'off')

    def test_shared_controls_are_preserved(self):
        cmd = missing_command(template(), 'skills_no_expansion', Path('m'), Path('a'))
        for k, v in COMMON.items():
            self.assertEqual(value(cmd, k), v)

    def test_duplicate_flag_is_not_silently_last_wins(self):
        for flag in ['--transport-limit', '--skill-level', '--candidate-expansion']:
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                missing_command(template() + [flag, '4'], 'skills_no_expansion', Path('m'), Path('a'))

    def test_free_budget_and_solver_scope_cannot_drift(self):
        for flag, bad in [('--transport-limit', '5'), ('--temporal-validation-seconds', '120'), ('--execution-area-policy', 'legacy_relative')]:
            cmd = template(); cmd[cmd.index(flag) + 1] = bad
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                missing_command(cmd, 'skills_no_expansion', Path('m'), Path('a'))

    def test_structured_preflights_cannot_differ(self):
        cmd = template(); cmd[cmd.index('--route-tail-preflight') + 1] = 'off'
        with self.assertRaises(ValueError):
            missing_command(cmd, 'skills_no_expansion', Path('m'), Path('a'))

    def test_completed_arms_are_not_scheduled_again(self):
        for arm in ['legacy_free', 'skills_with_expansion', 'unknown']:
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                missing_command(template(), arm, Path('m'), Path('a'))

    def test_preparation_never_marks_runtime_confirmation(self):
        for flag in ['--free-protection-confirmed', '--prepare-only']:
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                missing_command(template() + [flag], 'skills_no_expansion', Path('m'), Path('a'))


if __name__ == '__main__':
    unittest.main()
