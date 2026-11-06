"""未来路线与真实OBB足迹的首次相交弧长；不是动态可达性或规避证明。"""
import math


def route_geometry(points):
    """有限二维折线，去除连续重合点；缺图不能被解释成开放空间。"""
    result = []
    for row in points:
        point = tuple(float(x) for x in row[:2])
        if len(point) != 2 or not all(math.isfinite(x) for x in point):
            raise ValueError('invalid_route_point')
        if not result or math.dist(point, result[-1]) > 1e-6:
            result.append(point)
    if len(result) < 2:
        raise ValueError('insufficient_route_geometry')
    arc = [0.]
    for a, b in zip(result, result[1:]):
        arc.append(arc[-1] + math.dist(a, b))
    return result, arc


def project_ego(route, arc, position, max_distance=5.):
    """相交/回头路线存在多个不同时域的最近点时拒绝，不猜测自车在哪一圈。"""
    candidates = []
    for i, (a, b) in enumerate(zip(route, route[1:])):
        delta = [b[k]-a[k] for k in (0,1)]
        length = arc[i+1]-arc[i]
        ratio = max(0., min(1., sum((position[k]-a[k])*delta[k] for k in (0,1))/length**2))
        point = [a[k]+ratio*delta[k] for k in (0,1)]
        candidates.append((math.dist(position[:2], point), arc[i]+ratio*length))
    distance = min(x[0] for x in candidates)
    if distance > max_distance:
        raise ValueError('ego_too_far_from_route_for_clearance_certificate')
    nearest = [s for d,s in candidates if abs(d-distance) <= 1e-8]
    if max(nearest)-min(nearest) > 1e-3:
        raise ValueError('ambiguous_ego_route_projection')
    return min(nearest), distance


def route_box_intervals(route, arc, box):
    """用线段-有向矩形精确裁剪求中心线被足迹占用的弧长区间。"""
    x,y,yaw,length,width = [float(v) for v in box]
    if not all(math.isfinite(v) for v in (x,y,yaw,length,width)) or length <= 0 or width <= 0:
        raise ValueError('invalid_obstacle_box')
    c,s = math.cos(yaw), math.sin(yaw)
    def local(p):
        dx,dy = p[0]-x,p[1]-y
        return (dx*c+dy*s, -dx*s+dy*c)
    result = []
    for i,(a,b) in enumerate(zip(route,route[1:])):
        u,v = local(a),local(b)
        low,high = 0.,1.
        for axis,half in [(0,length/2), (1,width/2)]:
            delta = v[axis]-u[axis]
            if abs(delta) < 1e-12:
                if abs(u[axis]) > half:
                    low,high = 1.,0.
                    break
            else:
                entry,exit = sorted(((-half-u[axis])/delta,(half-u[axis])/delta))
                low,high = max(low,entry),min(high,exit)
        if low <= high:
            segment_length = arc[i+1]-arc[i]
            result.append([arc[i]+low*segment_length, arc[i]+high*segment_length])
    return result


def route_clearance(points, ego_position, boxes, minimum=15.):
    """整组至少一物体拦截路线；同地点多次经过、路线尾段等保守处理。"""
    try:
        route, arc = route_geometry(points)
        start,distance = project_ego(route,arc,ego_position)
        intervals = [item for box in boxes for item in route_box_intervals(route,arc,box)]
        if not intervals:
            return dict(accepted=False,reason='no_route_centerline_footprint_intersection')
        first = min(item[0] for item in intervals)
        clearance = first-start
        return dict(accepted=clearance >= minimum,
            reason=None if clearance >= minimum else 'route_footprint_entry_too_close_or_in_past',
            first_occupied_route_arc_m=first, ego_route_arc_m=start,
            ego_route_distance_m=distance, route_clearance_m=clearance, required_m=minimum,
            occupied_intervals=intervals, scope='route_centerline_occupancy_only_not_evasion_or_dynamic_safety')
    except (TypeError,ValueError,IndexError) as error:
        return dict(accepted=False,reason=str(error))
