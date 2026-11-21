"""正式20场的显式启动门禁；只接受同版本开发终态，不拼接或自动重试。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def prerequisites(protocol, development, audit):
    """开发只证明接线可用；正式20场仍需独立达到18场危险构建。"""
    reasons=[]
    if development.get('status')!='completed':reasons.append('development_not_completed')
    if audit.get('audit_issues'):reasons.append('development_integrity_failed')
    group=audit.get('groups',{}).get('skills_joint_event',{})
    summary=group.get('summary',{})
    rows=summary.get('rows',[])
    if summary.get('planned')!=3 or len(rows)!=3:reasons.append('development_three_scenes_incomplete')
    if summary.get('campaign_integrity_verified') is not True:reasons.append('development_not_verified')
    if group.get('audit_issues'):reasons.append('development_group_audit_failed')
    # 两个预先指定攻击样例需有完整危险证据，不要求第三个后期候选样例伪造成功。
    indexed={r['scenario_index']:r for r in rows}
    for scene in [6,74]:
        if indexed.get(scene,{}).get('effective_danger_constructed') is not True:
            reasons.append(f'development_danger_evidence_missing:{scene}')
    if indexed.get(6,{}).get('actual_static_exposure_frames',0)<=0:
        reasons.append('static_execution_attribution_unverified')
    if indexed.get(74,{}).get('actual_dynamic_frames',0)<=0:
        reasons.append('dynamic_execution_unverified')
    for row in rows:
        if row.get('audit_issues'):reasons.append('development_episode_audit_failed')
        if row.get('clean_execution') is not True:reasons.append('development_episode_not_clean')
    if len(protocol.get('planned_scenes',[]))!=20 or protocol.get('planned_attempts')!=20:
        reasons.append('formal_denominator_not20')
    if protocol.get('minimum_successes')!=18:reasons.append('formal_threshold_not18')
    return sorted(set(reasons))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    protocol=json.loads((args.protocol/'protocol.json').read_text())
    root=Path(protocol['workspace'])
    predecessor=Path(protocol['predecessor_experiment'])
    if not (predecessor/'launcher_result.json').exists():
        print(json.dumps(dict(allowed=False,reasons=['development_not_terminal'])))
        return 2
    sys.path[:0]=[str(root/'work/skills_danger_acceptance_20261104'),
                  str(root/'work/skills_modelpool_budget_dev_20261022')]
    from audit_campaign import audit
    from launch_pool_pair import verify_sources
    report=audit(predecessor,root)
    reasons=prerequisites(protocol,json.loads((predecessor/'launcher_result.json').read_text()),report)
    hashes=json.loads((args.protocol/'source_sha256.json').read_text())
    if not verify_sources(protocol,hashes)['verified']:reasons.append('formal_source_changed')
    old=json.loads((predecessor/'source_sha256.json').read_text())
    if any(name not in hashes or old[name]!=hashes[name] for name in old):
        reasons.append('development_runtime_source_differs')
    selection=Path(protocol['selection_file'])
    if hashlib.sha256(selection.read_bytes()).hexdigest()!=protocol['selection_sha256']:
        reasons.append('predeclared_selection_changed')
    frozen=json.loads(selection.read_text())
    manifest=json.loads((args.protocol/'manifest.json').read_text())
    for name in ['scenario_files','scenario_sha256']:
        if manifest[name]!=frozen[name]:reasons.append('formal_input_differs_from_selection')
    if protocol['planned_scenes']!=frozen['planned_scenes']:
        reasons.append('formal_scene_list_changed')
    print(json.dumps(dict(allowed=not reasons,reasons=sorted(set(reasons)),check_only=args.check_only)))
    if reasons:return 2
    if args.check_only:return 0
    # 只能显式进入，原控制器拒绝已有目录，失败后不覆盖/续拼。
    return subprocess.call([sys.executable,str(root/'work/skills_45m_event_20261104/run_stage.py'),
                            '--protocol',str(args.protocol)],cwd=root)


if __name__=='__main__':
    raise SystemExit(main())
