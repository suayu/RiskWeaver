"""新实验显式双轨入口；旧公共运行器及旧评分默认不改。"""
from pathlib import Path
import sys
import numpy as np
ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from dual_track_evaluation import VERSION,apply_result
from dual_track_pressure import DualObserver
from scene_fault_policy import is_fault,effect_population,install_continue


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
            max_frames=entry_steps,dual_track_protocol=VERSION)
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
        value['dual_track_scope']='conditional_online_legacy_pressure_components; exploratory_uncalibrated; not_cross_policy_common_exam'
        return value
    def overrides(*a,**kw):
        # 生成器保持旧双轨配置，EA 由独立观察器采集。
        return original_overrides(*a,**kw)+['sim.evaluation.ea.enabled=false']
    def episode(*a,**kw):
        result=original_episode(*a,**kw)
        # 比较入口已拥有并汇总实际观察器，历史入口只在独立运行时评分。
        if result.get('scoring_observer_protocol') != 'shared_DualObserver_stationary_v1':
            obs=DualObserver.latest
            result=apply_result(result,obs.challenge_result())
            result['danger_ttc_input_basis']=obs.danger_ttc_input_basis
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
        # 记录拆分口径，避免与历史控制序列版混用。
        if not entry_prepare:
            out=entry_output
            runner.dump_json(out/'dual_track_entry.json',dict(
                instruction=base.INSTRUCTION,skills=entry_skills,
                collision_in_challenge=False,off_route_in_challenge=False,failure_penalty_applied_once=True,
                dual_track_protocol=VERSION, outcome_basis='obb_ttc_feasible_composite',
                challenge_components=['source_obb_ttc','ea','reachability_compression','reachability_unsolvable'],
                challenge_minimum_coverage=.8, legacy_composite_d_enabled=True, historical_tca_not_used=True,
                generator_safety_unchanged=True))
    finally:
        restore_continue()
        runner.FormalRiskObserver=original_class;runner.run_episode=original_episode;runner.group_overrides=original_overrides;runner.aggregate_attempts=original_aggregate;runner.protocol_snapshot=original_protocol


if __name__=='__main__':main()
