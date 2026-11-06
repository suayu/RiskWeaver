"""验证无画像模式的可行性否决具有显式审计原因。"""
from pathlib import Path
p=Path('/home2/zhaoyx/scenario-dreamer/tests/test_multiagent_planner.py')
s=p.read_text(encoding='utf-8')
assert 'test_unprofiled_feasibility_veto_is_explicit' not in s
s+='''\n\ndef test_unprofiled_feasibility_veto_is_explicit():
    """两个有效角色意见不同时，不依赖角色名偶然排序。"""
    def generate(state,instructions,**kwargs):
        collaboration=json.loads(instructions[-1])
        if collaboration['role']=='proposer_feasibility':
            return {'attack':False,'reason':'avoidability_unverified'}
        return {'attack':True,'attack_target_id':2,'strategy':'cut_in'}
    planner=SimpleNamespace(generate_attack_plan=generate)
    engine=MultiAgentPlanner(settings('parallel_arbiter'))
    result=engine.run(planner,{'agents':[{'id':2}]},['original'])
    assert result=={'attack':False,'reason':'avoidability_unverified'}
    assert engine.last_trace['arbitration_reason']=='feasibility_veto'
    assert len(engine.last_trace['events'])==2
'''
p.write_text(s,encoding='utf-8')
