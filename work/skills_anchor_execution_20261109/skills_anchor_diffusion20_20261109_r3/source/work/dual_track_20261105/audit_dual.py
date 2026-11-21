"""只读双轨入口审计；无效评分保留null，不把接线通过当成指标已标定。"""
from pathlib import Path
import argparse,json,math


def audit(root,require_terminal=True):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());issues=[];groups={}
    if require_terminal and not (root/'launcher_result.json').is_file():raise RuntimeError('not_terminal')
    for arm in protocol['arms']:
        rows=[]
        for scene in protocol['planned_scenes']:
            path=root/arm/'movies'/f'scenario_{scene:03d}'/'attempt_result.json'
            if not path.is_file():rows.append(dict(scene=scene,attempted=False));continue
            result=json.loads(path.read_text());challenge=result.get('scene_challenge',{});value=challenge.get('score')
            if not challenge or challenge.get('collision_outcome_used') is not False or challenge.get('off_route_outcome_used') is not False:
                issues.append(f'{arm}:{scene}:challenge_outcome_contract')
            if value is not None and not 0<=value<=1:issues.append(f'{arm}:{scene}:invalid_range')
            for event in challenge.get('events',[]):
                e=event['challenge']
                if e.get('valid'):
                    expected=.5*e['control_space_loss']+.3*e['min_effort']/(1+e['min_effort'])+.2*e['fragility']
                    if not math.isclose(e['score'],expected,abs_tol=1e-9):issues.append(f'{arm}:{scene}:component_mismatch')
            ability=result.get('autonomous_driving_ability_score')
            if ability is not None:
                safe=result['completed'] and not result['collision'] and not result['off_route']
                expected=100*value*(1 if safe else .5*max(0,min(1,result['progress'])))
                if not math.isclose(ability,expected,abs_tol=1e-8):issues.append(f'{arm}:{scene}:ability_mismatch')
            rows.append(dict(scene=scene,attempted=True,frames=result.get('executed_steps'),challenge=value,
                challenge_coverage=challenge.get('coverage'),outcome=result.get('outcome_severity_score'),ability=ability,
                collision=result.get('collision'),completed=result.get('completed'),progress=result.get('progress')))
        groups[arm]=dict(planned=len(rows),attempted=sum(r['attempted'] for r in rows),rows=rows)
    return dict(wiring_verified=not issues,issues=issues,groups=groups,
        scientific_validation_complete=False,limitations=['finite_library','uncalibrated_weights','conditional_state_challenge'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=audit(a.root);raw=json.dumps(result,ensure_ascii=False,indent=2)
    if a.output:a.output.write_text(raw)
    else:print(raw)
