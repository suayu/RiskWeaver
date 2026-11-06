"""独立环境中的官方CommonRoad-Reach计算服务，不导入主仿真依赖。"""
import contextlib
import json
import sys
import time
import os
import ctypes


@contextlib.contextmanager
def native_stdout_to_stderr():
    """C++日志直接写文件描述符，必须与Python日志一同移出JSON协议通道。"""
    sys.stdout.flush()
    libc=ctypes.CDLL(None)
    libc.fflush(None)
    saved=os.dup(1)
    os.dup2(2,1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        libc.fflush(None)
        os.dup2(saved,1)
        os.close(saved)


def road_corridor(route, half_width, end_extension):
    """先合并整条路线的路面，再取补集；回折路线不能相互覆盖成禁行带。"""
    import numpy as np
    from shapely.geometry import LineString
    route=np.asarray(route,float)
    first=route[1]-route[0];first/=np.linalg.norm(first)
    last=route[-1]-route[-2];last/=np.linalg.norm(last)
    extended=np.vstack([route[0]-end_extension*first,route,route[-1]+end_extension*last])
    # 端部延伸允许初始/终点车体跨出路线采样端点，不能制造初始边界碰撞。
    road=LineString(extended).buffer(half_width,cap_style=2,join_style=1)
    domain=LineString(extended).buffer(half_width+10.,cap_style=2,join_style=1)
    return road,domain,domain.difference(road)


def compute(request):
    """内存转换场景、联合障碍占用与约束，运行官方C++多面体传播。"""
    import numpy as np
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    from shapely import constrained_delaunay_triangles
    from omegaconf import OmegaConf
    from commonroad.common.util import Interval
    from commonroad.geometry.shape import Rectangle, Polygon as CRPolygon, ShapeGroup
    from commonroad.scenario.scenario import Scenario
    from commonroad.scenario.lanelet import Lanelet
    from commonroad.scenario.state import InitialState, CustomState
    from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle, ObstacleType
    from commonroad.scenario.trajectory import Trajectory
    from commonroad.prediction.prediction import SetBasedPrediction, Occupancy
    from commonroad.planning.goal import GoalRegion
    from commonroad.planning.planning_problem import PlanningProblem
    from commonroad_reach.data_structure.configuration_builder import ConfigurationBuilder
    from commonroad_reach.data_structure.configuration import Configuration
    from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface
    from commonroad_reach.utility.configuration import create_curvilinear_coordinate_system
    from commonroad_clcs.util import resample_polyline

    if request['steps']<1 or request['dt']<=0:
        raise ValueError('invalid_time_horizon')
    if len(request['dynamic'])!=request['steps']+1:
        raise ValueError('incomplete_dynamic_time_coverage')
    if len({len(x) for x in request['dynamic']})!=1:
        raise ValueError('incomplete_dynamic_agent_coverage')
    start = time.monotonic()
    route = np.asarray(request['route'], float)[:, :2]
    route = route[np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1)>1e-6]]
    if len(route)<2:
        raise ValueError('invalid_route')
    route = resample_polyline(route, .5)
    direction = np.gradient(route, axis=0)
    direction /= np.linalg.norm(direction, axis=1)[:,None]
    normal = np.c_[-direction[:,1],direction[:,0]]
    half = request['lane_half_width']
    scenario = Scenario(dt=request['dt'])
    # 原项目没有真实道路边界；保留并明确标注相同路由中心走廊代理。
    scenario.add_objects(Lanelet(route+half*normal, route, route-half*normal, lanelet_id=1))
    ego = np.asarray(request['ego'], float)
    road,envelope,forbidden=road_corridor(route,half,max(5.,float(ego[5])))
    # 约束三角剖分保留内孔，不能把有效道路的孔洞填成禁行多边形。
    forbidden_triangles=constrained_delaunay_triangles(forbidden)
    shapes=[CRPolygon(np.asarray(poly.exterior.coords)[:-1]) for poly in forbidden_triangles.geoms
            if poly.area>1e-9]
    if shapes:
        scenario.add_objects(StaticObstacle(3000,ObstacleType.UNKNOWN,ShapeGroup(shapes),
            InitialState(position=np.zeros(2),orientation=0.,velocity=0.,time_step=0)))
    initial = InitialState(position=ego[:2], orientation=float(ego[4]),
        velocity=float(np.linalg.norm(ego[2:4])), acceleration=0., yaw_rate=0., slip_angle=0., time_step=0)
    goal = GoalRegion([CustomState(position=Rectangle(10., 2*half, route[-1]),
        time_step=Interval(0, request['steps']))])
    problem = PlanningProblem(100, initial, goal)
    # 官方曲线坐标转换不能处理所有顶点都在投影域外的大障碍。
    # 裁剪至比可行走廊更宽的计算域，保留所有可能与自车相交的部分。
    def occupancy_shape(c,y,l,w):
        poly=Polygon(Rectangle(float(l),float(w),np.asarray(c),float(y)).vertices)
        clipped=poly.intersection(envelope)
        if clipped.is_empty:return None
        pieces=[clipped] if clipped.geom_type=='Polygon' else list(clipped.geoms)
        shapes=[CRPolygon(np.asarray(piece.exterior.coords)[:-1]) for piece in pieces
                if piece.geom_type=='Polygon' and piece.area>1e-9]
        return ShapeGroup(shapes) if len(shapes)>1 else shapes[0] if shapes else None
    dynamic = request['dynamic']
    for row in range(len(dynamic[0])):
        shapes=[occupancy_shape(*obstacles[row]) for obstacles in dynamic]
        if not any(shape is not None for shape in shapes):continue
        occupancy=[Occupancy(k,shape) for k,shape in enumerate(shapes) if k>0 and shape is not None]
        # 预测占用使用世界坐标，初始变换为恒等；缺省只表示位于计算域外。
        shape0=shapes[0] or next(shape for shape in shapes if shape is not None)
        prediction=SetBasedPrediction(1,occupancy) if occupancy else None
        scenario.add_objects(DynamicObstacle(1000+row,ObstacleType.CAR,shape0,
            InitialState(position=np.zeros(2),orientation=0.,velocity=0.,time_step=0),prediction))
    for row, obs in enumerate(request['static']):
        shape=occupancy_shape(obs['center'],obs['yaw'],obs['length'],obs['width'])
        if shape is not None:
            scenario.add_objects(StaticObstacle(2000+row,ObstacleType.UNKNOWN,ShapeGroup([shape]) if isinstance(shape,CRPolygon) else shape,
                InitialState(position=np.zeros(2),orientation=0.,velocity=0.,time_step=0)))

    omega = OmegaConf.merge(ConfigurationBuilder().config_default,
        {'general':{'name_scenario':'RiskWeaver'},
         'planning':{'dt':request['dt'], 'steps_computation':request['steps'],
                     'coordinate_system':'CVLN', 'reference_point':'CENTER',
                     'uncertainty_p_lon':0.,'uncertainty_p_lat':0.,
                     'uncertainty_v_lon':0.,'uncertainty_v_lat':0.},
         'vehicle':{'ego':{'length':float(ego[5]),'width':float(ego[6]),
                     'v_lon_min':0.,'v_lon_max':request['max_speed'],
                     'v_lat_min':-request['max_lateral_speed'],'v_lat_max':request['max_lateral_speed'],
                     'a_lon_min':request['a_lon'][0],'a_lon_max':request['a_lon'][1],
                     'a_lat_min':request['a_lat'][0],'a_lat_max':request['a_lat'][1]}},
         'reachable_set':{'mode_computation':2, 'mode_inflation':request.get('inflation_mode',2),
                     'path_to_lut':request.get('lut_path','/home2/zhaoyx/scenario-dreamer/metadata/commonroad_reach_lut_20261014'),
                     'size_grid':request['grid_size'],'num_threads':1,
                     'prune_nodes_not_reaching_final_step':False}})
    config = Configuration(omega)
    extended = np.vstack([route[0]-30*direction[0],route,route[-1]+30*direction[-1]])
    clcs = create_curvilinear_coordinate_system(resample_polyline(extended,.5))
    config.update(scenario=scenario, planning_problem=problem, CLCS=clcs, list_ids_lanelets=[1])
    from commonroad_reach.data_structure.collision_checker import CollisionChecker
    original=CollisionChecker.retrieve_static_obstacles
    # 官方逐lanelet边界不等价于道路并集边界；本worker已有显式完整补集。
    # 只在隔离进程临时提供静态占据，官方C++传播、膨胀和碰撞计算保持原实现。
    CollisionChecker.retrieve_static_obstacles=staticmethod(lambda scene,network,traffic:
        list(scene.static_obstacles) if traffic else [s for s in scene.static_obstacles if s.obstacle_id==3000])
    try:
        interface = ReachableSetInterface(config)
        interface.compute_reachable_sets()
    finally:
        CollisionChecker.retrieve_static_obstacles=staticmethod(original)
    curve = []
    geometry = []
    for step in range(1, request['steps']+1):
        nodes = interface.reachable_set_at_step(step)
        polygons = [Polygon(np.asarray(node.position_rectangle.vertices)) for node in nodes]
        merged = unary_union(polygons)
        curve.append({'time_seconds':step*request['dt'],'area_m2':float(merged.area)})
        geometry.append({'time_seconds':step*request['dt'], 'polygon_count':len(polygons), 'union_bounds':list(merged.bounds) if not merged.is_empty else None})
    area = curve[-1]['area_m2']
    return {'area_m2':area, 'solvable':bool(area>0), 'horizon_seconds':request['dt']*request['steps'],
        'area_curve':curve, 'minimum_area_m2':min(x['area_m2'] for x in curve),
        'minimum_area_excludes_initial_state':True, 'backend':'commonroad_reach_2025.2.1',
        'area_basis':'union_of_curvilinear_position_rectangles_m2',
        'road_basis':'union_of_route_corridor_then_complement_v2', 'motion_model':'point_mass_outer_approximation',
        'ego_inflation':'three_circle_with_official_LUT' if request.get('inflation_mode',2)==3 else 'circumscribed_circle; conservative_vs_legacy_OBB',
        'ego_inflation_radius_m':config.vehicle.ego.radius_inflation,
        'road_geometry_audit':{'initial_center_in_road':road.covers(__import__('shapely.geometry',fromlist=['Point']).Point(ego[:2])),
                              'forbidden_overlaps_road_area_m2':forbidden.intersection(road).area},
        'node_counts':geometry,
        'constraints':{k:getattr(config.vehicle.ego,k) for k in ['v_lon_min','v_lon_max','v_lat_min','v_lat_max','a_lon_min','a_lon_max','a_lat_min','a_lat_max']}, 'compute_seconds':time.monotonic()-start}


def main():
    for line in sys.stdin:
        request = json.loads(line)
        try:
            # 官方诊断输出只能写stderr，stdout严格保留JSON响应协议。
            with native_stdout_to_stderr():
                result = compute(request)
            response = {'request_id':request['request_id'],'result':result}
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            response = {'request_id':request['request_id'],'error':type(error).__name__+': '+str(error)}
        print(json.dumps(response),flush=True)


if __name__ == '__main__':
    main()
