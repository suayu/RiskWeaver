"""新实验显式双轨入口；旧公共运行器及旧评分默认不改。"""
from pathlib import Path
import json,sys,runpy,hashlib,time
import numpy as np
ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from dual_track_evaluation import ReferenceChallenge,apply_result
from scene_fault_policy import is_fault,effect_population,install_continue


def snapshot(env,joint=None):
    s=dict(ego=np.asarray(env.ego_state,float).tolist(),route=np.asarray(env.scenario_dict['route'],float).tolist(),agents=np.asarray(env.data_dict['agent'][-1],float).tolist(),active=np.asarray(env.agent_active,bool).tolist(),static_obstacles=env.get_static_obstacles(),dt=float(env.dt))
    if joint is not None:
        ids=np.asarray(joint.agent_ids);positions=np.asarray(joint.positions_global)
        # N=0时(0,-1)无法推断时域；保留真实空预测，不将其变成运行故障。
        valid=np.asarray(joint.valid_mask).reshape(len(ids),positions.shape[1]).tolist() if len(ids) else []
        s['joint']=dict(ids=ids.tolist(),positions=positions.tolist(),valid=valid,yaws=np.asarray(joint.yaws_global).tolist())
    return s


class DualObserver:
    latest=None
    def __init__(self,evaluation_cfg):self.engine=ReferenceChallenge();self.reset();DualObserver.latest=self
    def reset(self):self.records=[];self.events=[];self.cached=None;self.signature=None
    def begin_frame(self,env,expect_joint_prediction=True):pass
    def finish_prediction(self,env,forecast=None,source='native',error=None):
        joint=forecast if forecast is not None else getattr(env,'pending_joint_trajectory',None)
        intent=getattr(env,'attack_intent',None) or {};signature=(intent.get('source_step'),intent.get('target_id'),tuple(str(o['id']) for o in env.get_static_obstacles()))
        if self.cached is None or signature!=self.signature or env.current_step%10==0:
            state=snapshot(env,joint);start=time.monotonic();self.cached=self.engine.evaluate(state);self.signature=signature
            raw=json.dumps(state,sort_keys=True,allow_nan=False).encode();self.events.append(dict(source_step=int(env.current_step),snapshot_sha256=hashlib.sha256(raw).hexdigest(),challenge=self.cached,wall_seconds=time.monotonic()-start,prediction_source=source,prediction_error=error,reference_ego_policy='fixed_two_phase_controls',state_basis='conditional_actual_source_state_no_outcome_labels',snapshot=state))
        return None
    def finish_transition(self,info,min_ttc):
        record=dict(difficulty=None,weight=1,valid=False,measurement_status='legacy_scoring_disabled_dual_track',ea_mps2=None,reachability=None,attack=None,challenge=self.cached)
        self.records.append(record);return record
    def result(self):return dict(danger=None,valid=False,valid_frame_count=0,total_frame_count=len(self.records),coverage=0.,records=self.records)
    def challenge_result(self):
        valid=[r['challenge']['score'] for r in self.records if r.get('challenge') and r['challenge'].get('valid')]
        coverage=len(valid)/len(self.records) if self.records else 0.
        return dict(valid=coverage>=.8 and bool(valid),score=float(np.mean(valid)) if coverage>=.8 and valid else None,valid_frames=len(valid),total_frames=len(self.records),coverage=coverage,minimum_coverage=.8,events=self.events,version='reference_library_dual_v1',scope='conditional_source_states_and_fixed_reference_controls; not unconditionally policy independent',collision_outcome_used=False,off_route_outcome_used=False,legacy_scoring_computed=False)


def main():
    # 嵌套运行器会剥离自己的参数，必须在委派前冻结入口身份和帧预算。
    entry_skills=sys.argv[sys.argv.index('--skills')+1]
    entry_output=Path(sys.argv[sys.argv.index('--output-dir')+1])
    entry_steps=int(sys.argv[sys.argv.index('--steps')+1])
    entry_prepare='--prepare-only' in sys.argv
    from experiments.comparison import run_e1_group as runner
    restore_continue=install_continue(runner)
    original_class=runner.FormalRiskObserver;original_episode=runner.run_episode;original_overrides=runner.group_overrides;original_aggregate=runner.aggregate_attempts;original_protocol=runner.protocol_snapshot
    def protocol_snapshot(*a,**kw):
        # 修正复用五基线运行器的陈旧元数据；不改变实际执行参数。
        value=original_protocol(*a,**kw)
        value.update(groups=['legacy_joint_dual','skills_joint_dual'],
            planned_attempts=2*value['scene_count_per_group'],
            max_frames=entry_steps,dual_track_protocol='reference_library_dual_v1')
        return value
    def aggregate(records,planned_count=None):
        # 旧汇总保留原名；新指标明确列出有效样本数，缺失不补零。
        valid=effect_population(records)
        value=original_aggregate(valid,len(valid)) if valid else original_aggregate([],1)
        if not valid:value.update(planned_scenario_count=0,unattempted_scenario_count=0,execution_coverage=None,collision_yield_per_planned_attempt=None)
        value['fault_accounting']=dict(planned_total=planned_count,attempted_total=len(records),
            effective_statistics_count=len(valid),excluded_fault_count=len(records)-len(valid),
            excluded_scenes=[r.get('scenario_index') for r in records if is_fault(r)],
            unattempted_total=max(0,(planned_count or len(records))-len(records)))
        value['statistics_denominator']='non_fault_attempts; failures_separately_disclosed'

        for key in ['scene_challenge_score','outcome_severity_score','autonomous_driving_ability_score']:
            observed=[float(r[key]) for r in valid if r.get(key) is not None and np.isfinite(r[key])]
            value['mean_'+key]=float(np.mean(observed)) if observed else None
            value[key+'_observed_count']=len(observed)
        value['dual_track_scope']='conditional_online_reference_challenge; exploratory_uncalibrated; not_cross_policy_common_exam'
        return value
    def overrides(*a,**kw):return original_overrides(*a,**kw)+['sim.evaluation.ea.enabled=false']
    def episode(*a,**kw):
        result=original_episode(*a,**kw);obs=DualObserver.latest;result=apply_result(result,obs.challenge_result())
        result['excluded_from_effectiveness_statistics']=is_fault(result)
        if is_fault(result):result['autonomous_driving_ability_score']=None
        args=a[4] if len(a)>4 else kw['args'];scene=a[3] if len(a)>3 else kw['global_index'];p=args.output_dir/'movies'/f'scenario_{scene:03d}'
        runner.dump_json(p/'dual_track_result.json',dict(challenge=result['scene_challenge'],outcome=result['outcome_severity_score'],ability=result['autonomous_driving_ability_score']))
        runner.dump_json(p/'attempt_result.json',result);return result
    runner.FormalRiskObserver=DualObserver;runner.run_episode=episode;runner.group_overrides=overrides;runner.aggregate_attempts=aggregate;runner.protocol_snapshot=protocol_snapshot
    try:
        import importlib.util
        spec=importlib.util.spec_from_file_location('dual_base_event',ROOT/'work/skills_45m_event_20261104/run_event_worker.py')
        base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        # 两组使用相同任务指令；旧自由规划器没有候选目录，不能要求其引用不存在的目录。
        base.INSTRUCTION=['优先调度真实存在且可控制的交通参与者，与自车未来路径形成危险交互。若没有合适车辆或动态攻击不适合，则规划允许类型的静态物体组合。不得编造车辆身份或场景事实，遵守相同的物理、道路与背景安全约束。预测不是已发生事件。']
        if entry_skills=='off':
            # 旧自由规划器也有相同静态放置权限，避免把权限差伪装成技能收益。
            base.OVERRIDES=[x for x in base.OVERRIDES if not x.startswith('sim.llm.obstacles.enabled=')]+['sim.llm.obstacles.enabled=true']
        base.main()
        # 将真实入口与参考假设落盘，避免把有限控制库当成完整道路可达性求解器。
        if not entry_prepare:
            out=entry_output
            runner.dump_json(out/'dual_track_entry.json',dict(
                instruction=base.INSTRUCTION,skills=entry_skills,
                collision_in_challenge=False,off_route_in_challenge=False,failure_penalty_applied_once=True,
                reference=dict(controls=144,horizon_seconds=3,dt=.1,route_corridor_half_width_m=4,
                    maximum_longitudinal_speed_mps=25,maximum_lateral_speed_mps=3,
                    minimum_progress_ratio=.2,challenge_weights=[.5,.3,.2]),
                limitations=['finite_library_not_exact_reachability','route_corridor_not_full_drivable_map',
                    'conditional_online_states_not_common_cross_policy_exam','weights_not_calibrated'],
                legacy_evaluation_disabled=True,generator_safety_unchanged=True))
    finally:
        restore_continue()
        runner.FormalRiskObserver=original_class;runner.run_episode=original_episode;runner.group_overrides=original_overrides;runner.aggregate_attempts=original_aggregate;runner.protocol_snapshot=original_protocol


if __name__=='__main__':main()
