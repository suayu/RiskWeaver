"""完整0.1–3秒事件网格的稀疏意图配方；不生成连续动作、不使用真实未来。"""
import numpy as np
from sparse_event import solve_sparse_event

TIMES=tuple(tick/10 for tick in range(1,31))


def screened_event_candidates(env,state,builder,semantic):
    """全量保存30时刻及拒绝；每个目标最多发布两条，原预检门槛不变。"""
    states=np.asarray(env.data_dict['agent'][-1],float)
    lanes=[np.asarray(line)[:,:2] for line in env.scenario_dict.get('lanes',[]) if len(line)>1]
    output=[];audit=[]
    for actor in state['agents']:
        target=actor['id']
        if isinstance(target,bool) or not isinstance(target,(int,np.integer)) or not 0<=target<len(states) or not env.agent_active[target]:
            raise ValueError('event_grid_requires_active_integer_actor_identity')
        accepted=[]
        for tick,event_time in enumerate(TIMES,1):
            estimate=solve_sparse_event(env.ego_state,states[target],event_time)
            record=dict(target_id=int(target),source_step=int(env.current_step),**estimate,published=False)
            audit.append(record)
            if not estimate['feasible']:continue
            candidate=dict(candidate_id=f'event_{target}_t{tick:02d}',target_id=int(target),strategy='lane_change',attack_family='merge_conflict',
                anchors=estimate['anchors'],duration=10,recipe='full_grid_source_CV_sparse_intercept_v1',
                event_time_seconds=event_time,event_reference='source_ego_constant_velocity_hypothesis',
                interpolation_event_residual_m=estimate['sparse_event_residual_m'],
                required_sparse_acceleration_norm_mps2=estimate['required_sparse_acceleration_norm_mps2'],
                event_within_current_lease=event_time<=10*env.dt,
                actual_collision_or_ego_reaction_certified=False,profile_match=None,profile_evidence_frames=0)
            metrics,reason=builder.screen(env,target,np.asarray(candidate['anchors']),lanes)
            record.update(candidate=candidate,screen_rejection=reason)
            if reason is not None:continue
            validation=semantic(state,candidate);record['semantic_validation']=validation
            if not validation['accepted']:continue
            candidate.update(**metrics,screened=True)
            accepted.append((candidate,record))
        # 固定双代表：最小稀疏加速度需求 + 最早通过时刻；不是看实验收益挑时刻。
        if accepted:
            ordered=[min(accepted,key=lambda pair:(pair[0]['required_sparse_acceleration_norm_mps2'],pair[0]['event_time_seconds'])),
                     min(accepted,key=lambda pair:pair[0]['event_time_seconds'])]
            seen=set()
            for candidate,record in ordered:
                if candidate['candidate_id'] in seen:continue
                seen.add(candidate['candidate_id']);record['published']=True;output.append(candidate)
    return output,audit
