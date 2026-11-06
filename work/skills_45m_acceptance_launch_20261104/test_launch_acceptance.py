"""正式20场启动门禁的边界反例，不使用API或仿真。"""
from launch_acceptance import prerequisites


def sample():
    protocol=dict(planned_scenes=list(range(20)),planned_attempts=20,minimum_successes=18)
    rows=[dict(scenario_index=i,effective_danger_constructed=i in [6,74],
               actual_static_exposure_frames=10 if i==6 else 0,
               actual_dynamic_frames=20 if i==74 else 0,audit_issues=[],clean_execution=True) for i in [6,74,181]]
    audit=dict(audit_issues=[],groups=dict(skills_joint_event=dict(audit_issues=[],
        summary=dict(planned=3,rows=rows,campaign_integrity_verified=True))))
    return protocol,dict(status='completed'),audit


def test_clean_development_allows_formal_test_only():
    assert prerequisites(*sample())==[]


def test_failed_development_not_accepted():
    p,d,a=sample();d['status']='failed'
    assert 'development_not_completed' in prerequisites(p,d,a)


def test_missing_static_attribution_cannot_pass():
    p,d,a=sample();a['groups']['skills_joint_event']['summary']['rows'][0]['actual_static_exposure_frames']=0
    assert 'static_execution_attribution_unverified' in prerequisites(p,d,a)


def test_collision_without_danger_attribution_not_enough():
    p,d,a=sample();row=a['groups']['skills_joint_event']['summary']['rows'][1]
    row.update(collision_any=True,effective_danger_constructed=False)
    assert 'development_danger_evidence_missing:74' in prerequisites(p,d,a)


def test_cannot_lower_formal_threshold():
    p,d,a=sample();p['minimum_successes']=17
    assert 'formal_threshold_not18' in prerequisites(p,d,a)


def test_missing_development_scene_not_ignored():
    p,d,a=sample();a['groups']['skills_joint_event']['summary']['rows'].pop()
    assert 'development_three_scenes_incomplete' in prerequisites(p,d,a)


def test_failure_in_non_attacking_scene_cannot_be_ignored():
    p,d,a=sample();a['groups']['skills_joint_event']['summary']['rows'][2]['clean_execution']=False
    assert 'development_episode_not_clean' in prerequisites(p,d,a)
