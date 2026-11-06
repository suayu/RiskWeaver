"""短程双轨接线门禁，解码全部视频；只写独立work收据。"""
from pathlib import Path
import hashlib,json,subprocess
from audit_dual import audit

ROOT=Path('/home2/zhaoyx/scenario-dreamer')


def main():
    root=ROOT/'experiments/dual_track_smoke_20261105_r1';report=audit(root);issues=list(report['issues']);videos=[]
    if json.loads((root/'launcher_result.json').read_text())['status']!='completed':issues.append('smoke_not_completed')
    if not json.loads((root/'runtime_source_integrity.json').read_text())['verified']:issues.append('runtime_source_changed')
    for name,expected in json.loads((root/'source_sha256.json').read_text()).items():
        if hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()!=expected:issues.append('snapshot_changed:'+name)
    for arm in ['legacy_joint_dual','skills_joint_dual']:
        scene=root/arm/'movies/scenario_015';result=json.loads((scene/'attempt_result.json').read_text())
        if result['executed_steps']!=20 or not result['scene_challenge']['valid']:issues.append(arm+':execution_or_challenge')
        for key in ['generation_failure','runtime_failure','planner_service_failure_count','prediction_service_failure_count']:
            if result.get(key):issues.append(arm+':'+key)
        if any(result['feasibility'].get(k,0) for k in ['background_collision_frames','background_static_collision_frames']):issues.append(arm+':background_collision')
        calls=[json.loads(x) for x in (root/(arm+'_api_calls.jsonl')).read_text().splitlines() if x.strip()]
        if not 0<len(calls)<=4 or any(c.get('model_requested')!='kimi-k3.1-pro' or c.get('model_returned')!='kimi-k3.1-pro' or c.get('error') for c in calls):issues.append(arm+':physical_model_budget')
        states=sum(json.loads(x).get('kind')=='state' for x in (scene/'execution_trace.jsonl').read_text().splitlines() if x.strip())
        for video in [scene/'scenario_015.mp4',root/(arm+'_integer_video_exports')/'scenario_015.mp4']:
            decoded=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,timeout=180)
            probe=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','json',str(video)])
            frames=int(json.loads(probe)['streams'][0]['nb_read_frames'])
            passed=decoded.returncode==0 and frames==states
            videos.append(dict(path=str(video.relative_to(root)),frames=frames,trace_states=states,verified=passed))
            if not passed:issues.append(str(video)+':decode_or_frame_mismatch')
    receipt=dict(allowed=not issues,issues=issues,dual_audit=report,videos=videos,
        scope='20frame_wiring_only_not_attack_or_score_scientific_validation',
        formal_instruction_shared_goal_corrected=True)
    (ROOT/'work/dual_track_20261105/smoke_gate.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps(dict(allowed=receipt['allowed'],issues=issues,videos=len(videos))))
    if issues:raise RuntimeError('smoke_gate_failed')


if __name__=='__main__':main()
