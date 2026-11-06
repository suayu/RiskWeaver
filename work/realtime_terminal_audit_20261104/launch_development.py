"""先验收真实单场闭环，再显式启动同源码开发组；不重试或改写前置实验。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home2/zhaoyx/scenario-dreamer')


def assess(report,launcher):
    reasons=[]
    if not report['terminal']:
        return ['predecessor_not_terminal']
    if launcher.get('status')!='completed' or not report['accounting_verified']:
        reasons.append('predecessor_failed_or_accounting_invalid')
    for arm,data in report['groups'].items():
        if launcher.get('groups',{}).get(arm,{}).get('exit_code')!=0:
            reasons.append(arm+':nonzero_or_missing_exit')
        if data['attempt_record_count']!=1 or data['executed_scenes']!=1:
            reasons.append(arm+':missing_executed_attempt')
        if data['highlevel_attack_scenes']!=1:
            reasons.append(arm+':no_accepted_highlevel_attack')
        for episode in data['episodes']:
            if episode.get('status')!='completed' or episode.get('runtime_failure') or episode.get('planner_service_failure_count'):
                reasons.append(arm+':not_clean_terminal')
            safety=episode.get('feasibility') or {}
            if any(safety.get(k) is None or safety[k]!=0 for k in ['background_collision_frames','background_static_collision_frames']):
                reasons.append(arm+':missing_or_failed_background_audit')
            if not episode.get('video_paths'):
                reasons.append(arm+':missing_video')
    return reasons


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    protocol=json.loads((args.protocol/'protocol.json').read_text())
    expected=protocol['predecessor_guard_sha256']
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=expected:
        raise ValueError('guard_not_frozen')
    sys.path[:0]=[str(ROOT/'work/realtime_terminal_audit_20261104'),
                  str(ROOT/'work/skills_modelpool_budget_dev_20261022')]
    from audit_realtime_pair import audit
    from launch_pool_pair import verify_sources
    predecessor=Path(protocol['predecessor_experiment'])
    if not (predecessor/'launcher_result.json').exists():
        print(json.dumps({'allowed':False,'reasons':['predecessor_not_terminal']}))
        return 2
    report=audit(predecessor)
    reasons=assess(report,json.loads((predecessor/'launcher_result.json').read_text()))
    hashes=json.loads((args.protocol/'source_sha256.json').read_text())
    if not verify_sources(protocol,hashes)['verified']:
        reasons.append('current_source_changed')
    if json.loads((predecessor/'source_sha256.json').read_text())!=hashes:
        reasons.append('development_and_smoke_source_differ')
    if not reasons:
        # 每个归档视频完整解码，主视频帧数必须等于含初态trace。
        for data in report['groups'].values():
            for episode in data['episodes']:
                for name in episode['video_paths']:
                    video=predecessor/name
                    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
                        '-show_entries','stream=nb_read_frames','-of','json',str(video)],capture_output=True,text=True)
                    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
                    if probe.returncode or decode.returncode:
                        reasons.append('video_decode_failed')
                        continue
                    frames=int(json.loads(probe.stdout)['streams'][0]['nb_read_frames'])
                    if frames!=episode['trace_states']:
                        reasons.append('video_trace_frame_count_mismatch')
    print(json.dumps({'allowed':not reasons,'reasons':reasons,'check_only':args.check_only}))
    if reasons:
        return 2
    if args.check_only:
        return 0
    return subprocess.call([sys.executable,str(ROOT/'work/rag_integration_20261030/run_dynamic_pair_20261104.py'),
                            '--protocol',str(args.protocol)],cwd=ROOT)


if __name__=='__main__':
    raise SystemExit(main())
