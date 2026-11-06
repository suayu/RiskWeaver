"""回折路线必须使用路面并集补集，不可把另一段合法路面判成禁行。"""
import numpy as np
from shapely.geometry import Point
from experiments.comparison.commonroad_worker import road_corridor


def test_hairpin_forbidden_region_never_overlaps_valid_road():
    route=np.array([[0.,0.],[0.,10.],[-10.,10.],[-10.,-10.]])
    road,domain,forbidden=road_corridor(route,4.,5.)
    assert road.covers(Point(0.,0.)) and not forbidden.covers(Point(0.,0.))
    assert forbidden.intersection(road).area==0.
    assert abs(domain.area-road.area-forbidden.area)<1e-6


def test_initial_vehicle_can_extend_behind_first_route_sample():
    road,_,forbidden=road_corridor([[0.,0.],[30.,0.]],4.,5.)
    assert road.covers(Point(-2.5,0.)) and not forbidden.covers(Point(-2.5,0.))
