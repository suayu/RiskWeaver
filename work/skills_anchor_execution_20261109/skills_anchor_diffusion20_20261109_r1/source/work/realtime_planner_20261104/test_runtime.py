import pytest
from work.realtime_planner_20261104.runtime import install
from work.realtime_planner_20261104.transport import RealtimeClient, MODEL

class Planner:
    def __init__(self):
        self.client=object(); self.provider='dashscope'; self.model_names=(MODEL,)

def test_off_preserves_constructor(tmp_path):
    before=Planner.__init__
    with install(tmp_path,False,Planner):
        assert Planner.__init__ is before
        assert not isinstance(Planner().client,RealtimeClient)

def test_on_installs_before_budget_wrapper_and_restores_on_error(tmp_path,monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY','test-secret')
    before=Planner.__init__
    with pytest.raises(RuntimeError):
        with install(tmp_path,True,Planner):
            assert isinstance(Planner().client,RealtimeClient)
            raise RuntimeError('downstream')
    assert Planner.__init__ is before
