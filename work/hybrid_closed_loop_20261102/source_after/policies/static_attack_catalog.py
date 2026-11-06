"""静态兜底的封闭组合目录；复用模板、15米边界和背景安全约束。"""
import copy
import itertools
import math
import numpy as np

from experiments.comparison.e1_metrics import polyline_distance
from policies.ego_profile import obstacle_box
from policies.joint_safety import obb_clearance
from policies.risk_metrics import RouteReachability


def geometric_grid(env, resolution=.5, half_width=1.75):
    """局部道路走廊栅格，仅计算几何未占用面积，不求自车规避路径或动态可达性。"""
    ego=np.asarray(env.ego_state,float)
    lanes=[np.asarray(line,float)[:,:2] for line in env.scenario_dict.get('lanes',[]) if len(line)>1]
    if not lanes:return dict(valid=False,reason='missing_lane_geometry')
    x,y=np.meshgrid(np.arange(0,60,resolution)+resolution/2,
                     np.arange(-12,12,resolution)+resolution/2)
    forward=np.array([math.cos(ego[4]),math.sin(ego[4])]);left=np.array([-forward[1],forward[0]])
    points=ego[:2]+x.reshape(-1,1)*forward+y.reshape(-1,1)*left
    distance=np.minimum.reduce([polyline_distance(points,line) for line in lanes])
    return dict(valid=True,points=points[distance<=half_width],resolution=resolution,
        lane_half_width=half_width,basis='local_0_to_60m_lane_corridor_minus_static_footprints',
        not_dynamic_reachability_or_evasion_witness=True)


def free_area(grid, obstacles):
    if not grid['valid']:return None
    points=grid['points'];occupied=np.zeros(len(points),dtype=bool)
    for item in obstacles:
        box=obstacle_box(item);c,s=math.cos(box[2]),math.sin(box[2]);delta=points-box[:2]
        along=delta@np.array([c,s]);side=delta@np.array([-s,c])
        occupied|=(abs(along)<=box[3]/2)&(abs(side)<=box[4]/2)
    return float(np.count_nonzero(~occupied)*grid['resolution']**2)


def materialize(planner, placements):
    groups=[]
    for index,placement in enumerate(placements):
        groups.append([item.to_public_dict() for item in planner.obstacle_catalog.materialize(
            placement,new_id=lambda name:f'static-candidate-{index}-{name}')])
    return groups


def validate_placements(env, planner, builder, placements, grid=None):
    """整组复核，不把过滤后残存的半个组合当成完整方案；不修改环境。"""
    if not 1<=len(placements)<=2:return dict(accepted=False,reason='static_group_count')
    ego=np.asarray(env.ego_state,float);angle=math.pi/2-ego[4]
    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    local=[dict(item,center=((np.asarray(item['center'])-ego[:2])@rotation.T).tolist(),
                yaw=float(item['yaw']+angle)) for item in placements]
    checked=planner._validate_obstacle_plan(local,planner._normalize_env_state(env.get_state_for_planning()))
    if len(checked)!=len(local):return dict(accepted=False,reason='original_obstacle_template_validation')
    groups=materialize(planner,placements);primitives=list(itertools.chain.from_iterable(groups))
    existing=env.get_static_obstacles()
    for group_a,group_b in itertools.combinations(groups,2):
        if any(obb_clearance(obstacle_box(a),obstacle_box(b))<0 for a in group_a for b in group_b):
            return dict(accepted=False,reason='combination_groups_overlap')
    if any(obb_clearance(obstacle_box(a),obstacle_box(b))<builder.margin for a in primitives for b in existing):
        return dict(accepted=False,reason='existing_static_clearance')
    states=np.asarray(env.data_dict['agent'][-1],float)
    states=states[np.asarray(env.agent_active,bool)] if states.ndim==2 else np.empty((0,7))
    minimum=float('inf');times=np.linspace(0,3,61)
    for state in states:
        position=state[:2]+times[:,None]*state[2:4]
        predicted=np.column_stack([position,np.full(len(times),state[4]),
                                   np.full(len(times),state[5]),np.full(len(times),state[6])])
        for primitive in primitives:
            box=obstacle_box(primitive)
            clearance=float(np.min(obb_clearance(predicted,np.broadcast_to(box,predicted.shape))))
            minimum=min(minimum,clearance)
            if clearance<builder.margin:return dict(accepted=False,reason='background_CV_clearance',margin_m=clearance)
    grid=grid if grid is not None else geometric_grid(env)
    remaining=free_area(grid,existing+primitives)
    if remaining is None or remaining<=0:
        return dict(accepted=False,reason='nonpositive_or_unknown_geometric_free_area',area_m2=remaining)
    return dict(accepted=True,reason=None,geometric_free_area_m2=remaining,
        geometric_area_basis=grid['basis'],area_resolution_m=grid['resolution'],
        dynamic_reachability_or_escape_not_certified=True,
        background_CV_margin_m=minimum if np.isfinite(minimum) else None,
        primitive_count=len(primitives))


def build_static_combinations(env,planner,builder,limit=24):
    """沿当前路线的未来位置生成多种已注册模板和两组组合，再交给LLM选择。"""
    if not planner.use_obstacles:return [],[dict(reason='static_generation_disabled')]
    try:route,arc,tangent=RouteReachability._route_geometry(env.scenario_dict['route'])
    except (ValueError,KeyError):return [],[dict(reason='invalid_route_geometry')]
    ego=np.asarray(env.ego_state,float);near=int(np.argmin(np.linalg.norm(route-ego[:2],axis=1)))
    start=arc[near];grid=geometric_grid(env)
    if not grid['valid']:return [],[dict(reason=grid['reason'])]
    variants=[('cone_line',3,1.2),('transverse_barrier',2,1.2),('debris_cluster',3,1.2),
        ('disabled_vehicle',1,1.),('construction_block',2,1.2),('construction_zone',1,3.),
        ('longitudinal_barrier',4,1.5)]
    proposals=[]
    for ahead in (20.,27.,35.):
        if start+ahead>arc[-1]:continue
        center,direction=RouteReachability._sample_route(route,arc,tangent,start+ahead)
        yaw=float(math.atan2(direction[1],direction[0]));left=np.array([-direction[1],direction[0]])
        for kind,count,spacing in variants:
            offsets=(0.,) if kind in ('construction_zone','disabled_vehicle') else (-.8,0.,.8)
            for offset in offsets:
                proposals.append(dict(type=kind,center=(center+offset*left).tolist(),yaw=yaw,count=count,spacing=spacing))
    singles=[];audit=[]
    for item in proposals:
        result=validate_placements(env,planner,builder,[item],grid)
        audit.append(dict(placements=[item],**result))
        if result['accepted']:singles.append((item,result))
    # 先发布多类型的组合，再补单模板；原模板内部多个水马/锥桶仍视为一组。
    combinations=[]
    for (a,_),(b,_) in itertools.combinations(singles,2):
        if a['type']==b['type'] or np.linalg.norm(np.asarray(a['center'])-b['center'])<3:continue
        result=validate_placements(env,planner,builder,[a,b],grid)
        audit.append(dict(placements=[a,b],**result))
        if result['accepted']:combinations.append(([a,b],result))
        if len(combinations)>=max(1,limit//2):break
    published=combinations+[([item],result) for item,result in singles]
    output=[];seen=set()
    for placements,result in published:
        types=tuple(sorted(item['type'] for item in placements))
        # 首轮覆盖不同模板类型，避免长目录全是同一种障碍物。
        if types in seen:continue
        seen.add(types)
        output.append(dict(obstacle_candidate_id=f'static_{len(output)}',placements=copy.deepcopy(placements),
            template_types=list(types),compatible_candidates=[],risk_basis='future_route_obstruction',**result))
        if len(output)>=limit:break
    return output,audit
