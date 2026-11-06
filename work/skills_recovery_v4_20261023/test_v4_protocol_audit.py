"""共同控制和回退包装器的负例；零模型调用、零GPU场景执行。"""
import copy
from types import SimpleNamespace as NS
import pytest
from omegaconf import OmegaConf
from audit_recovery_v4 import component_configuration
import launch_recovery_pair as entry


def fixture():
    cfg=OmegaConf.create(dict(sim=dict(evaluation=dict(reachability=dict(execution_gate_policy='positive_area')))))
    recovery=dict(execution_area_policy='positive_area',temporal_recovery='on',temporal_max_consecutive=1,
                  temporal_acceptance_budget_seconds=60.,route_tail_preflight='on',source_sha256={'a':'hash1'})
    video=dict(video_export='integer',fps=5,source_sha256={'b':'hash2'})
    return cfg,recovery,video,{'a':'hash1','b':'hash2'}


def test_frozen_common_controls_and_skills_only_preflight():
    cfg,r,v,h=fixture()
    assert component_configuration(cfg,r,v,'skills_full_shared_recovery',h)
    assert not component_configuration(cfg,r,v,'legacy_shared_recovery',h)
    r['route_tail_preflight']='off'
    assert component_configuration(cfg,r,v,'legacy_shared_recovery',h)


@pytest.mark.parametrize('which,value',[('temporal_acceptance_budget_seconds',120),('temporal_max_consecutive',2),
    ('execution_area_policy','legacy_relative'),('temporal_recovery','off'),('source_sha256',{})])
def test_missing_or_mismatched_controls_do_not_pass_audit(which,value):
    cfg,r,v,h=fixture();r[which]=value
    assert not component_configuration(cfg,r,v,'skills_full_shared_recovery',h)


def test_source_change_or_noninteger_export_is_detected():
    cfg,r,v,h=fixture();h['a']='changed'
    assert not component_configuration(cfg,r,v,'skills_full_shared_recovery',h)
    h['a']='hash1';v['video_export']='legacy'
    assert not component_configuration(cfg,r,v,'skills_full_shared_recovery',h)


def test_insufficient_disk_never_enters_launcher(monkeypatch):
    monkeypatch.setattr(entry.shutil,'disk_usage',lambda p:NS(free=5.49*2**30))
    monkeypatch.setattr(entry.launcher,'main',lambda:pytest.fail('must not launch before disk check'))
    with pytest.raises(RuntimeError,match='insufficient_free_disk'):entry.main()


def test_wrapper_installs_and_restores_actual_audit_hook(monkeypatch):
    monkeypatch.setattr(entry.shutil,'disk_usage',lambda p:NS(free=8*2**30))
    original=entry.pool_acceptance.audit_pool;idle=entry.launcher.idle_gpus
    def checked():
        assert entry.pool_acceptance.audit_pool is entry.audit_v4
        assert entry.launcher.idle_gpus is not idle
    monkeypatch.setattr(entry.launcher,'main',checked)
    entry.main()
    assert entry.pool_acceptance.audit_pool is original and entry.launcher.idle_gpus is idle
