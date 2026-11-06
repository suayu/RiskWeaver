"""为无画像记忆检索增加调用边界与模式隔离测试。"""
from pathlib import Path
p=Path('/home2/zhaoyx/scenario-dreamer/tests/test_multiagent_planner.py')
s=p.read_text(encoding='utf-8')
assert 'test_unprofiled_memory_preserves_original_planner_path' not in s
s=s.replace('from types import SimpleNamespace\n','from types import SimpleNamespace\nimport json\n',1)
s+='''\n\ndef test_unprofiled_memory_preserves_original_planner_path(tmp_path):
    """无画像检索只扩展协作提案，仍由原规划器验证离散计划。"""
    bank = dict(schema_version=1, planning_regime='unprofiled',
                source_scene_ids=['train.pkl'], records=[dict(policy_id='ego',
                scene_conditions={'road':'straight'}, scene_id='train.pkl',
                lesson='Recorded failure; danger unmeasured.', D=None)])
    path=tmp_path/'memory.json'
    path.write_text(json.dumps(bank),encoding='utf-8')
    seen=[]
    def generate(state,instructions,**kwargs):
        seen.append(instructions)
        return {'attack':True,'attack_target_id':2,'strategy':'cut_in'}
    planner=SimpleNamespace(generate_attack_plan=generate)
    engine=MultiAgentPlanner(SimpleNamespace(mode='parallel_memory',memory_path=str(path)))
    result=engine.run(planner,{'agents':[{'id':2}]},['original'],
                      memory_context={'policy_id':'ego','scene_id':'eval.pkl',
                                      'scene_conditions':{'road':'straight'}})
    assert result['attack'] and len(seen)==2
    assert all(instructions[0]=='original' for instructions in seen)
    assert engine.last_trace['memory_hits']==1
    assert all(len(instructions)==2 for instructions in seen)
    with pytest.raises(ValueError,match='overlaps'):
        engine.run(planner,{'agents':[]},['original'],memory_context={'policy_id':'ego','scene_id':'train.pkl'})
\n\ndef test_unprofiled_memory_rejects_profile_training_bank(tmp_path):
    """画像筛选失败经验不能直接充当无画像规划证据。"""
    path=tmp_path/'memory.json'
    path.write_text(json.dumps(dict(schema_version=1,source_scene_ids=[],records=[])),encoding='utf-8')
    engine=MultiAgentPlanner(SimpleNamespace(mode='parallel_memory',memory_path=str(path)))
    with pytest.raises(ValueError,match='regime mismatch'):
        engine.run(SimpleNamespace(),{'agents':[]},[],memory_context={'policy_id':'ego'})
'''
p.write_text(s,encoding='utf-8')
