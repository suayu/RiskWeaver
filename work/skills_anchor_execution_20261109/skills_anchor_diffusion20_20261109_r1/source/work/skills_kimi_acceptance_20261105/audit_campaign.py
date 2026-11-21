"""只读整组验收：固定场景分母，验证实际产物后才汇总危险构建证据。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from danger_acceptance import audit_episode, aggregate


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def video_check(video, expected_states=None):
    """完整解码而非仅打开视频；主视频须含每个trace状态。"""
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,timeout=180)
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=nb_read_frames','-of','json',str(video)],capture_output=True,timeout=180)
    frames=int(json.loads(probe.stdout)['streams'][0]['nb_read_frames']) if probe.returncode==0 else None
    return dict(path=str(video),frames=frames,verified=decode.returncode==0 and frames is not None
                and (expected_states is None or frames==expected_states))


def audit(root, workspace):
    """失败和未尝试保留；本工具不认证自然语言全文正确或攻击因果收益。"""
    protocol=read(root/'protocol.json')
    if not (root/'launcher_result.json').exists():
        raise RuntimeError('campaign_not_terminal')
    issues=[]
    if read(root/'launcher_result.json').get('status')!='completed':
        issues.append('launcher_not_completed')
    if read(root/'runtime_source_integrity.json').get('verified') is not True:
        issues.append('runtime_source_integrity_not_verified')
    # 验收已运行版本的快照，后续公共代码升级不会改写历史结果。
    frozen_sources=read(root/'source_sha256.json')
    if not frozen_sources or len(frozen_sources)!=protocol['source_count']:
        issues.append('source_manifest_count_mismatch')
    for name,expected in frozen_sources.items():
        snapshot=root/'source_snapshot'/name
        if not snapshot.is_file() or digest(snapshot)!=expected:
            issues.append('snapshot_mismatch:'+name)
    if digest(root/'manifest.json')!=protocol['manifest_sha256']:
        issues.append('manifest_sha_mismatch')
    manifest=read(root/'manifest.json')
    if manifest['original_scenario_indices']!=protocol['planned_scenes']:
        issues.append('planned_scene_order_mismatch')
    if len(manifest['scenario_files'])!=len(manifest['scenario_sha256']):
        issues.append('input_hash_count_mismatch')
    for filename,expected in zip(manifest['scenario_files'],manifest['scenario_sha256']):
        if not Path(filename).is_file() or digest(Path(filename))!=expected:
            issues.append('input_scene_sha_mismatch:'+filename)
    if len(manifest['scenario_files'])!=len(protocol['planned_scenes']):
        issues.append('manifest_scene_count_mismatch')
    sys.path.insert(0,str(workspace/'work/realtime_terminal_audit_20261104'))
    from audit_chat_accounting import audit as accounting_audit
    accounting=accounting_audit(root)
    if not accounting['accounting_verified']:
        issues.extend(accounting['audit_issues'])
    groups={}
    for arm in protocol['arms']:
        directory=root/arm
        entry=read(directory/'event_joint_entry.json')
        arm_issues=[]
        if entry.get('actual_config_verified') is not True:
            arm_issues.append('actual_configuration_not_verified')
        attempts={}
        for path in directory.glob('movies/*/attempt_result.json'):
            value=read(path);scene=value['scenario_index']
            if scene in attempts:raise ValueError('duplicate_scene_attempt')
            if scene not in protocol['planned_scenes']:raise ValueError('unexpected_scene_attempt')
            attempts[scene]=(path,value)
        reports=[];videos=[]
        for scene in protocol['planned_scenes']:
            if scene not in attempts:continue
            path,value=attempts[scene];trace=lines(path.parent/'execution_trace.jsonl')
            main_videos=list(path.parent.glob('*.mp4'))
            checks=[video_check(p,sum(r.get('kind')=='state' for r in trace)) for p in main_videos]
            videos.extend(checks)
            reports.append(audit_episode(value,trace,bool(checks) and all(c['verified'] for c in checks)))
        # 导出副本同样需完整解码，不以主视频通过掩盖损坏副本。
        main_paths={v['path'] for v in videos}
        for path in root.glob(arm+'_integer_video_exports/*.mp4'):
            if str(path) not in main_paths:videos.append(video_check(path))
        if not videos or not all(v['verified'] for v in videos):arm_issues.append('video_verification_failed')
        verified=not issues and not arm_issues
        groups[arm]=dict(summary=aggregate(protocol['planned_scenes'],reports,verified),
                         videos=videos,audit_issues=arm_issues)
    return dict(groups=groups,audit_issues=issues,accounting=accounting,
                scope='attributed_actual_exposure_and_OBB_CV_danger; not_causal_or_semantic_certificate')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--workspace',type=Path,default=Path('/home2/zhaoyx/scenario-dreamer'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        raise ValueError('output_must_not_modify_experiment')
    result=audit(args.root,args.workspace)
    with args.output.open('x',encoding='utf-8') as file:
        json.dump(result,file,ensure_ascii=False,indent=2)
    print(json.dumps(dict(audit_issues=result['audit_issues'],groups={
        k:{n:v['summary'][n] for n in ['planned','effective_danger_scenes','target_18_of20_met']}
        for k,v in result['groups'].items()})))
