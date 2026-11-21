"""新实验显式双轨入口；旧公共运行器及旧评分默认不改。"""
from pathlib import Path
import json,sys,runpy,hashlib,time
import numpy as np
ROOT=Path('/home2/zhaoyx/scenario-dreamer');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
from dual_track_evaluation import ReferenceChallenge,apply_result


def snapshot(env,joint=None):
    s=dict(ego=np.asarray(env.ego_state,float).tolist(),route=np.asarray(env.scenario_dict['route'],float).tolist(),agents=np.asarray(env.data_dict['agent'][-1],float).tolist(),active=np.asarray(env.agent_active,bool).tolist(),static_obstacles=env.get_static_obstacles(),dt=float(env.dt))
    if joint is not None:s['joint']=dict(ids=np.asarray(joint.agent_ids).tolist(),positions=np.asarray(joint.positions_global).tolist(),valid=np.asarray(joint.valid_mask).reshape(len(joint.agent_ids),-1).tolist(),yaws=np.asarray(joint.yaws_global).tolist())
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
    from experiments.comparison import run_e1_group as runner
    original_class=runner.FormalRiskObserver;original_episode=runner.run_episode;original_overrides=runner.group_overrides
    def overrides(*a,**kw):return original_overrides(*a,**kw)+['sim.evaluation.ea.enabled=false']
    def episode(*a,**kw):
        result=original_episode(*a,**kw);obs=DualObserver.latest;result=apply_result(result,obs.challenge_result())
        args=a[4] if len(a)>4 else kw['args'];scene=a[3] if len(a)>3 else kw['global_index'];p=args.output_dir/'movies'/f'scenario_{scene:03d}'
        runner.dump_json(p/'dual_track_result.json',dict(challenge=result['scene_challenge'],outcome=result['outcome_severity_score'],ability=result['autonomous_driving_ability_score']))
        runner.dump_json(p/'attempt_result.json',result);return result
    runner.FormalRiskObserver=DualObserver;runner.run_episode=episode;runner.group_overrides=overrides
    try:
        import importlib.util
        spec=importlib.util.spec_from_file_location('dual_base_event',ROOT/'work/skills_45m_event_20261104/run_event_worker.py')
        base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
        if '--skills' in sys.argv and sys.argv[sys.argv.index('--skills')+1]=='off':
            # 旧自由规划器也有相同静态放置权限，避免把权限差伪装成技能收益。
            base.OVERRIDES=[x for x in base.OVERRIDES if not x.startswith('sim.llm.obstacles.enabled=')]+['sim.llm.obstacles.enabled=true']
        base.main()
    finally:runner.FormalRiskObserver=original_class;runner.run_episode=original_episode;runner.group_overrides=original_overrides


if __name__=='__main__':main()
