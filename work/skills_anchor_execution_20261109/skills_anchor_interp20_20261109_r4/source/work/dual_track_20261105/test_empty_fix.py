"""复现空预测故障，并检查嵌套参数剥离后仍能落盘入口收据。"""
import sys,types,importlib.util
import numpy as np
import run_dual_worker_v3 as entry
from dual_track_evaluation import ReferenceChallenge


def test_zero_agents_joint_prediction():
    env=types.SimpleNamespace(ego_state=[0,0,10,0,0,4.5,2,1],
        scenario_dict={'route':[[i,0] for i in range(101)]},
        data_dict={'agent':[np.zeros((0,8))]},agent_active=np.zeros(0,bool),dt=.1,
        get_static_obstacles=lambda:[])
    joint=types.SimpleNamespace(agent_ids=np.empty(0,int),positions_global=np.empty((0,30,2)),
        valid_mask=np.empty((0,30),bool),yaws_global=np.empty((0,30)))
    state=entry.snapshot(env,joint)
    assert state['joint']['valid']==[]
    assert ReferenceChallenge().evaluate(state)['valid']


def test_nested_runner_can_remove_entry_arguments(monkeypatch,tmp_path):
    # 使用替身委派器模拟实际嵌套链对sys.argv的剥离，不请求SDK或运行仿真。
    runner=types.ModuleType('experiments.comparison.run_e1_group');saved={};outputs={}
    for name in ['FormalRiskObserver','run_episode','group_overrides','aggregate_attempts']:
        setattr(runner,name,lambda *a,**k: {})
    runner.protocol_snapshot=lambda *a,**k:{'scene_count_per_group':20}
    runner.dump_json=lambda p,v:outputs.update({p.name:v})
    saved.update(vars(runner))
    package=types.ModuleType('experiments.comparison');package.run_e1_group=runner
    monkeypatch.setitem(sys.modules,'experiments.comparison',package)
    monkeypatch.setitem(sys.modules,'experiments.comparison.run_e1_group',runner)
    base=types.ModuleType('fake_event');base.OVERRIDES=['sim.llm.obstacles.enabled=false']
    def delegate():
        outputs['protocol']=runner.protocol_snapshot()
        sys.argv=['delegated_worker']
    base.main=delegate
    spec=types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda m:None))
    monkeypatch.setattr(importlib.util,'spec_from_file_location',lambda *a:spec)
    monkeypatch.setattr(importlib.util,'module_from_spec',lambda s:base)
    monkeypatch.setattr(sys,'argv',['worker','--skills','off','--output-dir',str(tmp_path),'--steps','400'])
    entry.main()
    assert outputs['dual_track_entry.json']['skills']=='off'
    assert outputs['protocol']['planned_attempts']==40
    assert outputs['protocol']['max_frames']==400
    assert base.OVERRIDES==['sim.llm.obstacles.enabled=true']
    for key in ['run_episode','protocol_snapshot','aggregate_attempts']:
        assert getattr(runner,key) is saved[key]
