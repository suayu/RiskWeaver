"""画像联合模式的封闭障碍物候选：复用模板验证并检查与动态候选的兼容性。"""
from types import SimpleNamespace
import numpy as np

from policies.ego_profile import obstacle_box, project
from policies.joint_safety import obb_clearance


def build_obstacle_candidates(env, planner, dynamic_candidates, builder):
    """输出世界坐标模板；无道路、背景裕度或延迟制动规避见证时不提供候选。"""
    if not getattr(planner,'use_obstacles',False):
        return []
    ego=np.asarray(env.ego_state,float)
    route=np.asarray(env.scenario_dict['route'],float)[:,:2]
    forward=np.array([np.cos(ego[4]),np.sin(ego[4])])
    side=np.array([-forward[1],forward[0]])
    angle=np.pi/2-ego[4]
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    normalized=planner._normalize_env_state(env.get_state_for_planning())
    states=np.asarray(env.data_dict['agent'][-1],float)[np.asarray(env.agent_active,bool)]
    lanes=[np.asarray(l)[:,:2] for l in env.scenario_dict.get('lanes',[]) if len(l)>1 and project(np.asarray(l)[0,:2],l) is not None]
    if not lanes:
        return []
    times=np.linspace(0,3,61)
    speed=float(np.linalg.norm(ego[2:4]))
    braking=np.minimum(np.maximum(times-.4,0),speed/4)
    distance=speed*np.minimum(times,.4)+speed*braking-2*braking**2
    witness=ego[:2]+distance[:,None]*forward
    witness_boxes=np.column_stack((witness,np.full(len(times),ego[4]),np.full(len(times),ego[5]),np.full(len(times),ego[6])))
    if np.max(project(witness,route))+ego[6]/2>builder.half_width:
        return []
    existing=env.get_static_obstacles()
    output=[]
    for ahead in (max(18.,speed*.4+speed**2/8+5), max(25.,speed*.4+speed**2/8+10)):
        desired=ego[:2]+ahead*forward
        center=route[np.argmin(np.linalg.norm(route-desired,axis=1))]
        if not 16<(center-ego[:2])@forward<45:
            continue
        for offset in (0.,1.2,-1.2):
            location=center+offset*side
            placement=dict(type='cone_line',center=location.tolist(),yaw=float(ego[4]),count=1,spacing=1.)
            local=dict(placement,center=((location-ego[:2])@rotation.T).tolist(),yaw=float(ego[4]+angle))
            if not planner._validate_obstacle_plan([local],normalized):
                continue
            primitives=[x.to_public_dict() for x in planner.obstacle_catalog.materialize(placement,new_id=lambda name:f'candidate-{name}')]
            box=obstacle_box(primitives[0])
            if min(float(project(location,l)) for l in lanes)+max(box[3:])/2>builder.half_width:
                continue
            if any(obb_clearance(box,obstacle_box(o))<builder.margin for o in existing):
                continue
            safe=True
            margin=float('inf')
            for state in states:
                pos=state[:2]+times[:,None]*state[2:4]
                boxes=np.column_stack((pos,np.full(len(times),state[4]),np.full(len(times),state[5]),np.full(len(times),state[6])))
                clearance=float(np.min(obb_clearance(boxes,np.broadcast_to(box,boxes.shape))))
                margin=min(margin,clearance)
                safe &= clearance>=builder.margin
                safe &= np.min(obb_clearance(witness_boxes,boxes))>=.3
            if not safe or np.min(obb_clearance(witness_boxes,np.broadcast_to(box,witness_boxes.shape)))<.3:
                continue
            # 静态候选可独立使用；联合选择还须通过候选轨迹与新障碍物的配对筛查。
            proxy=SimpleNamespace(ego_state=env.ego_state,data_dict=env.data_dict,agent_active=env.agent_active,
                                  scenario_dict=env.scenario_dict,get_static_obstacles=lambda:existing+primitives)
            compatible=[c['candidate_id'] for c in dynamic_candidates if builder.screen(proxy,c['target_id'],c['anchors'],lanes)[1] is None]
            output.append(dict(obstacle_candidate_id=f'o{len(output)}',placement=placement,compatible_candidates=compatible,
                               background_margin_m=margin if np.isfinite(margin) else None,
                               avoidability='0.4s delay then 4m/s2 braking witness',risk_basis='forward route obstruction'))
    return output


def chosen_obstacle(response, context, ranking):
    """只允许目录中的单组模板，禁止模型自行摆放或选择不兼容组合。"""
    selected=response.get('obstacle_candidate_id')
    if selected is None:
        return []
    item=next((c for c in context.get('obstacle_candidates',[]) if c['obstacle_candidate_id']==selected),None)
    if item is None or (ranking and ranking[0] not in item['compatible_candidates']):
        raise ValueError('invalid_or_incompatible_obstacle_candidate')
    return [item['placement']]
