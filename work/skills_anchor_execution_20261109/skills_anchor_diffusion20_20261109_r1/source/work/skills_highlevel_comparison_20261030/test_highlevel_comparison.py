"""真正skills关闭机制与稳定skills入口的同初态比较；SDK替身不联网。"""
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util,json,sys
import numpy as np
import pytest

ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('legacy_skills_runner',HERE/'frozen_planning.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def fixture():
    spec=importlib.util.spec_from_file_location('event_prepare_fixture',ROOT/'work/skills_event_grid_20261030/prepare_planning.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(ROOT/'experiments/skills_temporal_development10_paired_20261029_r1/skills_full_shared_recovery/resolved_config.yaml')
    return next(s for s in module.samples() if s['scene']==74),cfg


def factory(sample,mode,invalid=False):
    from policies.llm_adversarial_planner import LLMAdversarialPlanner
    sys.path.insert(0,str(ROOT/'work/skills_scene_facts_prototype_20261028'))
    from test_fact_runtime import answer
    def create(*args,**kwargs):
        if mode=='legacy_off':
            c=next(c for c in sample['contexts']['temporal_original']['candidates'] if c['candidate_id']=='a0_8')
            ego=np.asarray(sample['state']['ego_state']);theta=np.pi/2-ego[4]
            rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            anchors=(np.asarray(c['anchors'])-ego[:2])@rotation.T
            if invalid:anchors[0]+=[10,0]
            value=dict(attack=True,attack_target_id=c['target_id'],strategy=c['strategy'],anchors=anchors.tolist(),reason='SDK fixture')
        else:
            payload=json.loads(kwargs['messages'][-1]['content'].rsplit('\n',1)[1]);value=answer(payload)
        return NS(model=kwargs['model'],choices=[NS(message=NS(content=json.dumps(value),reasoning_content=None),finish_reason='stop')],usage=NS(model_dump=lambda:dict(total_tokens=0)))
    return lambda **kwargs:LLMAdversarialPlanner(client=NS(chat=NS(completions=NS(create=create))),**kwargs)


@pytest.mark.parametrize('mode',['legacy_off','temporal_skills_common'])
def test_real_mechanisms_compile_same_source_without_event_grid(tmp_path,mode):
    sample,cfg=fixture();r=runner.run_trial(sample,mode,['deepseek-v4.1-flash'],cfg,tmp_path/mode,planner_factory=factory(sample,mode))
    assert r['accepted_plan'] and not r['planner_service_failure'] and r['budget']['transport_calls']==1
    assert not r['event_grid_enabled'] and r['new_execution_frames']==0 and r['D'] is None
    if mode=='legacy_off':
        assert r['selected_candidate'] is None and 'common_identity_evidence' not in r['planner_trace']
        assert len(r['planner_trace']['requests'])==1
    else:assert r['planner_trace']['common_identity_evidence']['response_identity_not_rewritten']


def test_original_legacy_validation_rejects_bad_anchor_not_posthoc_repaired(tmp_path):
    sample,cfg=fixture();r=runner.run_trial(sample,'legacy_off',['deepseek-v4.1-flash'],cfg,tmp_path/'bad',planner_factory=factory(sample,'legacy_off',True))
    assert not r['accepted_plan'] and not r['planner_service_failure'] and r['budget']['transport_calls']==1
    assert r['validation_reason'] and '第一个锚点' in r['validation_reason']


def test_legacy_arm_has_no_skills_engine():
    sample,cfg=fixture();assert runner.engine_for('legacy_off',cfg,sample) is None
