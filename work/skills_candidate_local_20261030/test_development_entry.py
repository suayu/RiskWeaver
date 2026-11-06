"""真实自由规划/skills装配的SDK替身回归；与实际模型额度状态解耦。"""
from pathlib import Path
import importlib.util,sys
import pytest
from omegaconf import OmegaConf

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_development as new
spec=importlib.util.spec_from_file_location('r3_test_support',ROOT/'work/skills_highlevel_comparison_20261030/test_highlevel_comparison.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)


@pytest.mark.parametrize('mode',new.ARMS)
def test_three_actual_mechanisms_no_network_no_input_mutation(tmp_path,mode):
    sample,cfg=support.fixture();before=OmegaConf.to_container(cfg,resolve=True)
    engine=new.engine_for(mode,cfg,sample)
    if mode=='legacy_off':assert engine is None
    else:assert engine.grounding_validation_mode==('global' if mode=='temporal_global' else 'candidate_local')
    # 完全离线替身的虚构模型名，不会恢复已耗尽模型，也不会产生真实HTTP。
    result=new.base.run_trial(sample,mode,['fixture-bailian-model'],cfg,tmp_path/mode,
        planner_factory=support.factory(sample,'legacy_off' if mode=='legacy_off' else 'temporal_skills_common'))
    assert result['accepted_plan'] and not result['planner_service_failure'] and result['budget']['transport_calls']==1
    assert result['D'] is None and result['new_execution_frames']==0 and before==OmegaConf.to_container(cfg,resolve=True)
    assert (result['planner_trace'].get('candidate_grounding_validation') is not None)==(mode=='temporal_candidate_local')


def test_legacy_original_bad_anchor_still_rejected(tmp_path):
    sample,cfg=support.fixture()
    result=new.base.run_trial(sample,'legacy_off',['fixture-bailian-model'],cfg,tmp_path/'invalid',
        planner_factory=support.factory(sample,'legacy_off',True))
    assert not result['accepted_plan'] and '第一个锚点' in result['validation_reason']


def test_unknown_arm_cannot_silently_enable_policy():
    with pytest.raises(ValueError):new.engine_for('all_pass',None,None)
