"""可回退入口与配置根目录恢复；不执行模拟或API。"""
from types import SimpleNamespace as NS
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import run_incremental as entry


class EntryTests(unittest.TestCase):
    def test_off_preserves_previous_entry_and_requires_recipe_off(self):
        with patch.object(entry.previous, 'main', return_value='old') as old:
            self.assertEqual(entry.run(NS(incremental_skills='off', active_window_recipes='off'), ['--skills','off']), 'old')
            old.assert_called_once_with()
        with self.assertRaises(ValueError):entry.run(NS(incremental_skills='off', active_window_recipes='on'), [])

    def test_on_requires_compatible_pipeline(self):
        with self.assertRaisesRegex(ValueError, 'explicit_pipeline'):
            entry.run(NS(incremental_skills='on', active_window_recipes='off'), ['--skills','off'])

    def test_config_root_is_final_and_factory_restores(self):
        original = lambda:['old_root']
        runner = NS(group_overrides=original)
        restored = []
        undo = entry.install_root(lambda *args:lambda:restored.append(True), None, runner)
        self.assertEqual(runner.group_overrides()[-1], 'sim.llm.skills.root='+str(entry.HERE/'skills'))
        undo()
        self.assertIs(runner.group_overrides, original)
        self.assertEqual(restored, [True])

    def test_existing_output_never_receives_new_protocol_on_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder);(path/'original.txt').write_text('keep')
            with patch.object(sys, 'argv', ['entry','--output-dir',folder]):
                with self.assertRaises(FileExistsError):entry.main()
            self.assertEqual([p.name for p in path.iterdir()], ['original.txt'])

    def test_recipe_switch_cannot_silently_do_nothing(self):
        required=['--skills','on','--compact-view','on','--event-budget','on',
                  '--reachability-preflight','on','--route-tail-preflight','on','--candidate-expansion','off']
        with self.assertRaisesRegex(ValueError,'recipes_require_candidate_expansion'):
            entry.run(NS(incremental_skills='on',active_window_recipes='on'),required)


if __name__ == '__main__':unittest.main()
