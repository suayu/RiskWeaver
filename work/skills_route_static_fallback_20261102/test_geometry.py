"""精确足迹入口、直道等价、弯道、路线自交及刚体变换；无API。"""
import math
import unittest

from geometry import route_clearance


class GeometryTest(unittest.TestCase):
    def test_straight_measures_front_footprint_not_center(self):
        route = [[0,0],[100,0]]
        self.assertFalse(route_clearance(route,[0,0],[[16,0,0,4,2]])['accepted'])
        self.assertEqual(route_clearance(route,[0,0],[[17,0,0,4,2]])['route_clearance_m'],15)

    def test_curved_route_has_far_arc_with_small_current_heading_projection(self):
        result = route_clearance([[0,0],[10,0],[10,100]],[0,0],[[10,20,math.pi/2,4,2]])
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['route_clearance_m'],28)

    def test_rejects_close_initial_turn(self):
        result = route_clearance([[0,0],[10,0],[10,100]],[0,0],[[10,3,math.pi/2,4,2]])
        self.assertFalse(result['accepted'])

    def test_centerline_miss_is_not_obstruction(self):
        self.assertFalse(route_clearance([[0,0],[100,0]],[0,0],[[30,8,0,4,2]])['accepted'])

    def test_missing_map_and_route_tail_reject(self):
        self.assertFalse(route_clearance([], [0,0], [[30,0,0,4,2]])['accepted'])
        self.assertFalse(route_clearance([[0,0],[10,0]],[0,0],[[30,0,0,4,2]])['accepted'])

    def test_invalid_dimensions_and_nan_are_not_accepted(self):
        for box in [[30,0,0,0,2],[30,0,float('nan'),4,2]]:
            self.assertFalse(route_clearance([[0,0],[100,0]],[0,0],[box])['accepted'])

    def test_self_intersection_ego_anchor_is_unknown(self):
        result = route_clearance([[-10,-10],[10,10],[-10,10],[10,-10]],[0,0],[[8,-8,0,2,2]])
        self.assertEqual(result['reason'],'ambiguous_ego_route_projection')

    def test_past_route_occupation_is_not_future_safe_placement(self):
        result = route_clearance([[0,0],[100,0]],[40,0],[[20,0,0,4,2]])
        self.assertFalse(result['accepted'])

    def test_rigid_transform_preserves_result(self):
        angle=.73
        def move(p):
            return [p[0]*math.cos(angle)-p[1]*math.sin(angle)+100,
                p[0]*math.sin(angle)+p[1]*math.cos(angle)-60]
        route=[[0,0],[10,0],[10,100]]; box=[10,20,math.pi/2,4,2]
        a=route_clearance(route,[0,0],[box])
        b=route_clearance([move(p) for p in route],move([0,0]),[move(box[:2])+[box[2]+angle,*box[3:]]])
        self.assertEqual(a['accepted'],b['accepted'])
        self.assertAlmostEqual(a['route_clearance_m'],b['route_clearance_m'])

    def test_all_group_primitives_contribute_to_first_intersection(self):
        result=route_clearance([[0,0],[100,0]],[0,0],[[40,0,0,4,2],[5,0,0,1,1]])
        self.assertFalse(result['accepted'])


if __name__ == '__main__':
    unittest.main()
