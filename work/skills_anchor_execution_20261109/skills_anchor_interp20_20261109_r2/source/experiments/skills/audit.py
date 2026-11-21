"""离线验收技能对照：保留预定分母，区分标记执行、意图跟随和真实风险。"""
import argparse
import json
from pathlib import Path
import math
import shutil
import subprocess


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def distance(a,b):
    return math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))


def probe_video(path):
    """读取真实视频流及帧数，不把存在的空文件视为完整视频。"""
    executable=shutil.which('ffprobe')
    if not executable:return dict(path=str(path),valid=False,reason='ffprobe_unavailable')
    try:
        result=subprocess.run([executable,'-v','error','-select_streams','v:0','-count_frames',
            '-show_entries','stream=nb_read_frames,width,height','-of','json',str(path)],
            capture_output=True,text=True,timeout=30,check=True)
        stream=json.loads(result.stdout)['streams'][0]
        return dict(path=str(path),valid=stream['width']>0 and stream['height']>0,
                    frame_count=int(stream['nb_read_frames']))
    except (subprocess.SubprocessError,ValueError,KeyError,IndexError):
        return dict(path=str(path),valid=False,reason='video_probe_failed')


def audit_attempt(path):
    """只读已落盘事实；不把缺失产物或D=null补成成功或零风险。"""
    record=read_json(path);directory=path.parent
    trace=directory/'execution_trace.jsonl'
    rows=[];malformed=0
    if trace.exists():
        for line in trace.read_text(encoding='utf8').splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:malformed+=1
    states={r['step']:r for r in rows if r['kind']=='state'}
    outputs=[r for r in rows if r['kind']=='llm_output']
    frames=[r for r in rows if r['kind']=='attack_execution' and r.get('executed')]
    predictions={r['step']:r for r in rows if r['kind']=='execution_prediction'}
    plans=[r for r in rows if r['kind']=='prediction' and r.get('attack_intent')]
    first=[];seen=set()
    # 每个意图只评估首个输出未来，不能将反复滚动的32点当独立样本。
    for r in plans:
        intent=r['attack_intent'];source=intent.get('source_step',r['step'])
        identity=(source,intent['target_id'])
        if identity in seen:continue
        seen.add(identity)
        joint=(predictions.get(r['step']) or {}).get('joint')
        if not joint or intent['target_id'] not in joint['agent_ids']:continue
        actor=joint['agent_ids'].index(intent['target_id'])
        positions=joint['positions_global'][actor]
        forecast_source=joint['source_step']
        state=states.get(r['step']);errors=[];approach=[]
        for second,anchor in enumerate(intent['anchors']):
            if second==0:continue
            index=source+second*10-forecast_source-1
            if not 0<=index<len(positions):continue
            point=positions[index]
            item=dict(anchor_time_seconds=second,predicted_error_m=distance(point,anchor))
            actual=states.get(source+second*10)
            if actual and intent['target_id']<len(actual['agents']):
                item['actual_error_m']=distance(actual['agents'][intent['target_id']],anchor)
                item['within_attack_validity']=second*10<=intent.get('duration',10)
            if state:
                ego=state['ego'];future=(forecast_source+index+1-r['step'])*.1
                approach.append(distance(point,[ego[0]+future*ego[2],ego[1]+future*ego[3]]))
            errors.append(item)
        first.append(dict(source_step=source,target_id=intent['target_id'],strategy=intent['strategy'],
                          duration_frames=intent.get('duration',10),anchor_errors=errors,
                          predicted_aligned_ego_min_distance_m=min(approach) if approach else None))
    api_calls=0;schema_rejections=0;no_candidate=0;responses=0;models={};calls_wall=0.
    for r in outputs:
        t=r.get('planner_trace') or {};calls_wall+=float(r.get('planner_wall_seconds',0.))
        if t.get('mode')=='skills_candidate_ranking':
            count=max(0,int(t.get('api_calls_total',0))-int(t.get('api_calls_before',0)))
            schema_rejections+=len(t.get('schema_rejections',[]))
            no_candidate+=int(t.get('plan_validation')=='no_feasible_candidates')
            responses+=sum('response' in request for request in t.get('requests',[]))
        else:
            count=len(t.get('attempted_models',[]))
            responses+=int(bool(t.get('successful_model')))
        api_calls+=count
        if count and t.get('successful_model'):
            name=t['successful_model'];models[name]=models.get(name,0)+1
    executed=record.get('executed_steps',0)
    videos=[probe_video(p) for p in directory.glob('*.mp4')]
    completeness=dict(trace_exists=trace.exists(),malformed_trace_lines=malformed,
        state_count=len(states),terminal_state_present=executed in states,
        state_sequence_complete=set(range(executed+1))<=set(states),
        video_count=len(videos),video_checks=videos,
        video_frames_match=bool(videos) and all(v['valid'] and v.get('frame_count')==executed+1 for v in videos))
    continuous_flags=sum(bool((r.get('joint') or {}).get('metadata',{}).get('anchor_guidance_active'))
                         for r in predictions.values())
    return dict(scenario_index=record.get('scenario_index'),status=record.get('status'),
        executed_steps=executed,attack_plan_count=record.get('attack_plan_count',0),
        actual_attack_frames=len(frames),attack_executed=bool(frames),
        anchor_guidance_forecast_frames=continuous_flags,first_forecasts=first,
        api_calls=api_calls,http_response_events=responses,schema_rejections=schema_rejections,
        no_candidate_queries=no_candidate,planning_queries=len(outputs),planning_wall_seconds=calls_wall,
        successful_model_events=models,collision=record.get('collision'),off_route=record.get('off_route'),
        completed=record.get('completed'),progress=record.get('progress'),
        D=record.get('scenario_danger_score'),D_valid_frames=record.get('danger_valid_frame_count',0),
        D_total_frames=record.get('danger_total_frame_count',0),
        EA=record.get('mean_evasive_acceleration_mps2'),OBB_TTC=record.get('min_ttc_s'),
        background_collision_frames=sum(bool(s.get('background_pairs')) for s in states.values()),
        background_static_collision_frames=sum(bool(s.get('background_static_pairs')) for s in states.values()),
        static_obstacles_seen=any(bool(s.get('static_obstacles')) for s in states.values()),
        generation_failure=bool(record.get('generation_failure')),
        generation_failure_reason=record.get('generation_failure_reason'),
        runtime_failure=bool(record.get('runtime_failure')),
        planner_service_failures=record.get('planner_service_failure_count',0),
        completeness=completeness,attempt_result=str(path))


def audit_experiment(root):
    manifest=read_json(root/'manifest.json');launch=read_json(root/'launch.json')
    arms=[a[0] if isinstance(a,list) else a for a in launch['arms']]
    planned=manifest['scenario_count'];summary={}
    for name in arms:
        attempts=[audit_attempt(p) for p in sorted((root/name).rglob('attempt_result.json'))]
        indices=[a['scenario_index'] for a in attempts]
        duplicates=len(indices)-len(set(indices))
        valid_frames=sum(a['D_valid_frames'] for a in attempts)
        total_frames=sum(a['D_total_frames'] for a in attempts)
        attacks=sum(a['attack_executed'] for a in attempts)
        complete=len(attempts)==planned and not duplicates
        bg=sum(a['background_collision_frames']+a['background_static_collision_frames'] for a in attempts)
        artefacts=all(a['completeness']['video_frames_match'] and a['completeness']['state_sequence_complete']
                      and a['completeness']['terminal_state_present']
                      and not a['completeness']['malformed_trace_lines'] for a in attempts)
        final_files=list((root/name).rglob('exit_status.json'))
        arm_terminal=bool(final_files) and all('exit_code' in read_json(p) for p in final_files)
        summary[name]=dict(planned_scenes=planned,attempted_scenes=len(attempts),
            unattempted_scenes=max(0,planned-len(attempts)),duplicate_scenes=duplicates,
            actual_execution_scenes=sum(a['executed_steps']>0 for a in attempts),
            actual_attack_scenes=attacks,actual_attack_coverage=attacks/planned,
            actual_attack_frames=sum(a['actual_attack_frames'] for a in attempts),
            attack_plans=sum(a['attack_plan_count'] for a in attempts),
            api_calls=sum(a['api_calls'] for a in attempts),
            planner_wall_seconds=sum(a['planning_wall_seconds'] for a in attempts),
            collisions=sum(a['collision'] is True for a in attempts),
            collision_yield=sum(a['collision'] is True for a in attempts)/planned,
            D_valid_scene_count=sum(a['D'] is not None for a in attempts),
            D_mean_valid_scenes=(sum(a['D'] for a in attempts if a['D'] is not None)/sum(a['D'] is not None for a in attempts)) if any(a['D'] is not None for a in attempts) else None,
            D_frame_coverage=valid_frames/total_frames if total_frames else None,
            generation_failures=sum(a['generation_failure'] for a in attempts),
            runtime_failures=sum(a['runtime_failure'] for a in attempts),
            planner_service_failures=sum(a['planner_service_failures'] for a in attempts),
            background_collision_frames=bg,arm_terminal=arm_terminal,attempts=attempts,
            all_planned_attempted=complete,artifacts_complete=complete and artefacts,
            execution_coverage_gate_pass=complete and arm_terminal and attacks>planned/2 and bg==0 and artefacts,
            scope_note='Coverage gate does not establish low reaction margin or anchor fidelity; inspect those separately.')
    return dict(experiment=str(root),groups=summary,all_terminal=all(s['arm_terminal'] for s in summary.values()),
                forecast_time_assumption='dt=0.1s; anchors at0/1/2/3s; report original plan validity separately')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.experiment.resolve() in args.output.resolve().parents:
        raise ValueError('audit output must be outside immutable experiment')
    result=audit_experiment(args.experiment)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({name:{k:v for k,v in value.items() if k!='attempts'} for name,value in result['groups'].items()},ensure_ascii=False))


if __name__=='__main__':main()
