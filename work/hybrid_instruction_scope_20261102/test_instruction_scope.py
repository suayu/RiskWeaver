"""指令隔离、异常恢复及SDK日志审计；无真实请求。"""
import copy
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

from instruction_scope import INSTRUCTION, audit_calls, install


class InstructionTest(unittest.TestCase):
    def test_install_and_restore_old_dynamic_instruction(self):
        runner = NS(INSTRUCTION=['禁止放置静态障碍物'])
        seen = []
        def original(options, target):
            seen.append(copy.deepcopy(target.INSTRUCTION))
            return lambda: seen.append('restored')
        module = NS(install_variant=original)
        with install(module):
            undo = module.install_variant(NS(), runner)
            self.assertEqual(runner.INSTRUCTION, INSTRUCTION)
            self.assertEqual(seen[0], ['禁止放置静态障碍物'])
            undo()
            self.assertEqual(runner.INSTRUCTION, ['禁止放置静态障碍物'])
        self.assertIs(module.install_variant, original)

    def test_restoration_even_if_nested_restore_raises(self):
        runner = NS(INSTRUCTION=['old'])
        def broken():
            raise ValueError('fixture')
        module = NS(install_variant=lambda *args: broken)
        with install(module):
            undo = module.install_variant(NS(), runner)
            with self.assertRaises(ValueError):
                undo()
            self.assertEqual(runner.INSTRUCTION, ['old'])

    def test_audit_actual_payload_and_keep_invalid_calls(self):
        def call(value):
            return dict(messages=[dict(content='prefix\n'+json.dumps(dict(instruction=value)))])
        result = audit_calls([call(INSTRUCTION), call(['禁止放置静态障碍物']), {}])
        self.assertEqual([r['matches'] for r in result['rows']], [True, False, False])
        self.assertFalse(result['all_match'])
        self.assertFalse(audit_calls([])['all_match'])

    def test_scope_exit_restores_even_without_explicit_undo(self):
        runner = NS(INSTRUCTION=['old'])
        restored = []
        original = lambda *args: lambda: restored.append(True)
        module = NS(install_variant=original)
        with install(module):
            module.install_variant(NS(), runner)
        self.assertEqual(runner.INSTRUCTION, ['old'])
        self.assertEqual(restored, [True])
        self.assertIs(module.install_variant, original)

    def test_receipt_write_error_restores_old_instruction(self):
        runner = NS(INSTRUCTION=['old'])
        module = NS(install_variant=lambda *args: lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'file'
            file.write_text('occupied')
            with install(module, output=file):
                with self.assertRaises(OSError):
                    module.install_variant(NS(), runner)
                self.assertEqual(runner.INSTRUCTION, ['old'])


if __name__ == '__main__':
    unittest.main()
