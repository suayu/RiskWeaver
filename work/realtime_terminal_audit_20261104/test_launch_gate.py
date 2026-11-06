import copy
from launch_development import assess

def complete():
    e=dict(status='completed',feasibility=dict(background_collision_frames=0,background_static_collision_frames=0),video_paths=['v.mp4'])
    d=dict(attempt_record_count=1,executed_scenes=1,highlevel_attack_scenes=1,episodes=[e])
    return dict(terminal=True,accounting_verified=True,groups={'old':d}),dict(status='completed',groups={'old':dict(exit_code=0)})

def test_pending_never_passes():
    r,l=complete();r['terminal']=False
    assert assess(r,l)==['predecessor_not_terminal']

def test_clean_preconditions_pass():
    r,l=complete();assert not assess(r,l)

def test_no_attack_does_not_pass_with_healthy_process():
    r,l=complete();r['groups']['old']['highlevel_attack_scenes']=0
    assert any('no_accepted' in x for x in assess(r,l))

def test_missing_safety_is_not_zero():
    r,l=complete();r['groups']['old']['episodes'][0]['feasibility']={}
    assert any('background_audit' in x for x in assess(r,l))
