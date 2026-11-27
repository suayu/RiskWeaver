
from pathlib import Path
import os, sys, runpy
from contextlib import ExitStack
import numpy as np
ROOT=Path.cwd()
for name in ['skills_natural_rollback_20261121','skills_anchor_execution_20261109','skills_incremental_evidence_20261025','dual_track_20261105']:
    sys.path.insert(0,str(ROOT/'work'/name))
sys.path.insert(0,str(ROOT))
from experiments.comparison import run_e1_group as runner
from simulator import Simulator
from natural_worker import without_frame_cap
from json_safe import plain
from dual_track_pressure import DualObserver
from dual_track_evaluation import apply_result
from scene_fault_policy import is_fault
arm=os.environ['RW_COMPARISON_ARM']
for key in ('RW_NL_INSTRUCTION','RW_NL_PREFERENCE_ENABLED'):
    os.environ.pop(key,None)
raw_dump=runner.dump_json
runner.dump_json=lambda path,value:raw_dump(path,plain(value))
runner.FormalRiskObserver=DualObserver
natural,_=without_frame_cap(runner.run_episode,'count < args.steps')
Simulator.step,_=without_frame_cap(Simulator.step,'self.t == self.cfg.sim.steps')
original_aggregate=runner.aggregate_attempts
def aggregate(records,planned=None):
    result=original_aggregate(records,planned)
    valid=[r for r in records if not is_fault(r)]
    for key in ('scene_challenge_score','outcome_severity_score'):
        values=[float(r[key]) for r in valid if r.get(key) is not None and np.isfinite(r[key])]
        result[key+'_mean']=float(np.mean(values)) if values else None
        result[key+'_observed_count']=len(values)
    result['non_fault_count']=len(valid)
    return result
runner.aggregate_attempts=aggregate
def episode(env,policy,index,global_index,args,*a,**kw):
    # 同一自然终止和评分观察器用于每组；原执行故障仍单列。
    DualObserver.latest=None
    out=args.output_dir
    runner.dump_json(out/'comparison_entry.json',dict(arm=arm,frame_cap=None,
        allocation_horizon=env.steps,traffic_backend=env.traffic_backend,
        tilt=getattr(env.cfg.sim.behaviour_model,'tilt',None),
        skills=bool(env.cfg.sim.llm.skills.enabled),rag=bool(env.cfg.sim.rag.enabled),
        scoring='shared_DualObserver',reachability=env.cfg.sim.evaluation.reachability.backend))
    assert not env.cfg.sim.rag.enabled
    assert bool(env.cfg.sim.llm.skills.enabled)==(arm=='skills')
    if arm in ('skills','no_skills'):
        assert env.traffic_backend=='external_joint' and env.cfg.sim.llm.attack_mode=='joint'
    if arm=='scenario_dreamer':
        assert env.cfg.sim.behaviour_model.tilt==-10
    if os.environ.get('RW_COMPARISON_PREFLIGHT')=='1':
        raise SystemExit(0)
    result=natural(env,policy,index,global_index,args,*a,**kw)
    obs=DualObserver.latest
    result=apply_result(result,obs.challenge_result())
    result['danger_ttc_input_basis']=obs.danger_ttc_input_basis
    result['excluded_from_effectiveness_statistics']=is_fault(result)
    runner.dump_json(out/'movies'/f'scenario_{global_index:03d}'/'attempt_result.json',result)
    return result
runner.run_episode=episode
if arm=='skills':
    # 使用前次已验证的完整 Skills 入口，复用独立预算与静态兜底。
    entry=ROOT/'work/skills_anchor_execution_20261109/run_anchor_worker.py'
    sys.argv[0]=str(entry)
    runpy.run_path(str(entry),run_name='__main__')
else:
    original=runner.group_overrides
    def overrides(*args,**kwargs):
        values=original(*args,**kwargs)
        patch={
            'sim.llm.skills.enabled':'false','sim.rag.enabled':'false',
            'sim.evaluation.reachability.backend':'commonroad',
            'sim.evaluation.ea.enabled':'false',
        }
        if arm=='no_skills':
            patch.update({'sim.llm.attack_mode':'joint','sim.llm.obstacles.enabled':'true',
                          'sim.llm.provider':'deepseek','sim.llm.skills.static_fallback_enabled':'false'})
        return [v for v in values if v.split('=')[0] not in patch]+[f'{k}={v}' for k,v in patch.items()]
    runner.group_overrides=overrides
    if arm=='no_skills':
        from run_anchor_worker import install
        from anchor_execution import window_steps
        from policies import trajectory_projection as projection, evaluation_trace
        from policies.skills_planner import SkillsAttackPlanner
        from policies.joint_safety import NoSafeJointCandidate
        from policies.llm_adversarial_planner import LLMAdversarialPlanner
        import active_window_recipes
        original_plan=LLMAdversarialPlanner.generate_attack_plan
        def generate(self,*a,**kw):
            # 原生规划器已经按 0、1、2、3 秒返回四个锚点；同步其执行时长。
            plan=original_plan(self,*a,**kw)
            if isinstance(plan,dict) and plan.get('attack') is True:
                plan=dict(plan,duration=window_steps(.1))
            return plan
        LLMAdversarialPlanner.generate_attack_plan=generate
        runner.INSTRUCTION=['优先使用合适的交通参与者构造动态测试事件；动态不适用时考虑合法静态障碍物组合。']
        with install('interpolation',runner,SkillsAttackPlanner,runner.AdversarialScenarioGenerator,
                     runner.Simulator,active_window_recipes,projection,NoSafeJointCandidate,
                     witness_accel=4.,trace_module=evaluation_trace):
            runner.main()
    else:
        runner.main()
