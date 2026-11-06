"""四组完整性不能由两主组通过替代；中间组预检配置按协议而非名字判断。"""
import copy
from pathlib import Path
import tempfile
import unittest
from omegaconf import OmegaConf
from audit_four_arm import four_arm_gate, declared_configuration, smoke_prerequisite


class FourArmTests(unittest.TestCase):
    def test_middle_group_failure_prevents_gate_passing(self):
        groups={name:dict(group_integrity_pass=True,qualified_scenes=count) for name,count in [
            ('legacy_shared_recovery',2),('catalog_no_knowledge',1),('skills_no_expansion',2),('skills_full_shared_recovery',8)]}
        self.assertTrue(four_arm_gate(groups,groups)['passed'])
        bad=copy.deepcopy(groups);bad['catalog_no_knowledge']['group_integrity_pass']=False
        self.assertFalse(four_arm_gate(bad,groups)['passed'])
        missing=copy.deepcopy(groups);del missing['skills_no_expansion']
        self.assertFalse(four_arm_gate(missing,groups)['passed'])
        self.assertFalse(four_arm_gate({},groups)['passed'])

    def test_majority_and_comparison_cannot_be_relaxed_by_extra_groups(self):
        groups={name:dict(group_integrity_pass=True,qualified_scenes=count) for name,count in [
            ('legacy_shared_recovery',7),('catalog_no_knowledge',10),('skills_no_expansion',10),('skills_full_shared_recovery',6)]}
        self.assertFalse(four_arm_gate(groups,groups)['passed'])
        groups['legacy_shared_recovery']['qualified_scenes']=2
        groups['skills_full_shared_recovery']['qualified_scenes']=5
        self.assertFalse(four_arm_gate(groups,groups)['passed'])

    def test_middle_arm_must_match_own_declared_flags(self):
        cfg=OmegaConf.create(dict(sim=dict(llm=dict(skills=dict(enabled=True,level='none',candidate_expansion_enabled=False,event_budget_enabled=True)))))
        expected=dict(skills_enabled=True,level='none',candidate_expansion=False,active_recipes=False)
        receipt=dict(incremental_skills='on',active_window_recipes='off',bailian_capabilities='observed')
        self.assertTrue(declared_configuration(cfg,expected,receipt))
        cfg.sim.llm.skills.level='full'
        self.assertFalse(declared_configuration(cfg,expected,receipt))

    def test_missing_smoke_is_not_a_pass_or_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'not_started'
            result=smoke_prerequisite(path,'incremental_full',74)
            self.assertFalse(result['passed'])
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
