"""终态只读验收：故障台账、有效分母、配对交集与视频完整性。"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')
sys.path[:0]=[str(ROOT),str(ROOT/'work/skills_danger_acceptance_20261104')]
from danger_acceptance import audit_episode
from audit_campaign import video_check
from effect_summary import summarize,paired
from scene_fault_policy import is_fault


def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(root):
    if not (root/'launcher_result.json').exists():raise RuntimeError('wait_for_terminal_no_partial_conclusions')
    protocol=read(root/'protocol.json');issues=[];groups={};all_attempts={};all_reports={}
    frozen=read(root/'source_sha256.json')
    for name,expected in frozen.items():
        p=root/'source_snapshot'/name
        if not p.is_file() or sha(p)!=expected:issues.append('source_snapshot:'+name)
    if not read(root/'runtime_source_integrity.json').get('verified'):issues.append('runtime_source_integrity')
    manifest=read(root/'manifest.json')
    if sha(root/'manifest.json')!=protocol['manifest_sha256']:issues.append('manifest_integrity')
    for name,expected in zip(manifest['scenario_files'],manifest['scenario_sha256']):
        if sha(Path(name))!=expected:issues.append('input_scene_integrity:'+name)
    if manifest['original_scenario_indices']!=protocol['planned_scenes']:issues.append('scene_order')
    for arm in protocol['arms']:
        directory=root/arm;attempts=[];reports={};videos=[];arm_issues=[]
        calls=lines(root/(arm+'_api_calls.jsonl'));counts=Counter(c.get('scenario_index') for c in calls)
        if any(n>4 for n in counts.values()) or any(s not in protocol['planned_scenes'] for s in counts):arm_issues.append('physical_request_budget_or_identity')
        if any(c.get('model_requested') not in protocol['models'] or (c.get('model_returned') and c['model_returned'] not in protocol['models']) for c in calls):arm_issues.append('unapproved_model')
        for scene in protocol['planned_scenes']:
            p=directory/'movies'/f'scenario_{scene:03d}'/'attempt_result.json'
            if not p.exists():continue
            result=read(p);attempts.append(result);trace=lines(p.parent/'execution_trace.jsonl')
            states=sum(r.get('kind')=='state' for r in trace)
            checks=[video_check(v,states) for v in p.parent.glob('*.mp4')];videos.extend(checks)
            if not is_fault(result):
                if not checks or not all(v['verified'] for v in checks):arm_issues.append(f'{scene}:video_missing_or_invalid')
                c=result.get('scene_challenge') or {}
                if c.get('collision_outcome_used') is not False or c.get('off_route_outcome_used') is not False:arm_issues.append(f'{scene}:challenge_contract')
                ability=result.get('autonomous_driving_ability_score');score=result.get('scene_challenge_score')
                if ability is not None:
                    safe=result['completed'] and not result['collision'] and not result['off_route']
                    expected=100*score*(1 if safe else .5*max(0,min(1,result['progress'])))
                    if not math.isclose(ability,expected,abs_tol=1e-8):arm_issues.append(f'{scene}:ability_formula')
            reports[scene]=audit_episode(result,trace,bool(checks) and all(v['verified'] for v in checks))
        for v in root.glob(arm+'_integer_video_exports/*.mp4'):videos.append(video_check(v))
        summary=summarize(protocol['planned_scenes'],attempts,reports)
        groups[arm]=dict(summary=summary,episodes=reports,videos=videos,audit_issues=arm_issues,
            models=dict(Counter(c.get('model_returned') or 'no_success_response' for c in calls)),
            physical_calls=len(calls),physical_calls_per_scene=dict(counts),
            total_sdk_seconds=sum(c.get('elapsed_seconds',0) for c in calls),
            tokens=sum((c.get('usage') or {}).get('total_tokens',0) for c in calls))
        all_attempts[arm]=attempts;all_reports[arm]=reports
    left,right=protocol['arms']
    return dict(artifact_issues=issues,groups=groups,
        paired=paired(all_attempts[left],all_attempts[right],all_reports[left],all_reports[right]),
        goal_complete=False,score_calibration_proven=False,semantic_correctness_certified=False,
        note='faults_excluded_per_user; planned/fault/unattempted_disclosed; normal_no_attack_retained; skill_payload_audit_required_separately')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.resolve().is_relative_to(a.root.resolve()):raise ValueError('output_must_be_outside_experiment')
    result=audit(a.root);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v['summary'] for k,v in result['groups'].items()},ensure_ascii=False))
