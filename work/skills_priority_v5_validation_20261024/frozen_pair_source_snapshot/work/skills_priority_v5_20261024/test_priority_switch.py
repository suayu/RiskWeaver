"""检查回退开关与知识根覆盖，确保不会悄悄切换旧组。"""
from types import SimpleNamespace
from unittest.mock import patch
import sys
import unittest
import run_priority


class SwitchTests(unittest.TestCase):
    def test_default_off_calls_old_entry_with_original_arguments(self):
        args = ['entry','--output-dir','unused','--skills','off']
        with patch.object(sys,'argv',args), patch.object(run_priority.base,'main',return_value=17) as main:
            self.assertEqual(run_priority.main(),17)
            self.assertEqual(sys.argv,args)
            main.assert_called_once_with()

    def test_priority_on_requires_explicit_compatible_pipeline(self):
        with patch.object(sys,'argv',['entry','--priority-skills','on','--skills','on']):
            with self.assertRaises(ValueError): run_priority.main()

    def test_registry_root_is_final_and_factory_restores(self):
        original = lambda:['old']
        runner = SimpleNamespace(group_overrides=original)
        def install(options,runner):
            runner.group_overrides = lambda:['installed','sim.llm.skills.root=old']
            return lambda:setattr(runner,'group_overrides',original)
        restore = run_priority.priority_override(install,None,runner)
        self.assertEqual(runner.group_overrides()[-1],'sim.llm.skills.root='+str(run_priority.HERE/'skills'))
        restore();self.assertIs(runner.group_overrides,original)


if __name__ == '__main__':unittest.main()
