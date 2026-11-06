"""插值事件、物理上限与刚体不变性回归；零API/仿真。"""
import unittest
import numpy as np
from sparse_event import solve_sparse_event


class SparseEventTest(unittest.TestCase):
    def test_fractional_event_hits_sparse_interpolation_not_only_quadratic(self):
        e=[0,0,0,1];s=[-2,2,0,-1]
        r=solve_sparse_event(e,s,1.5)
        self.assertTrue(r['feasible']);self.assertLess(r['sparse_event_residual_m'],1e-12)
        a=np.asarray(r['sparse_acceleration_mps2']);quadratic=np.asarray(s[:2])+1.5*np.asarray(s[2:4])+.5*1.5**2*a
        self.assertGreater(np.linalg.norm(quadratic-np.asarray(r['desired_ego_CV_position_world'])),.01)

    def test_infeasible_time_not_silently_clipped_into_success(self):
        r=solve_sparse_event([0,0,0,1],[-20,20,0,-1],.5)
        self.assertFalse(r['feasible']);self.assertNotIn('anchors',r)

    def test_hard_cap_cannot_be_increased(self):
        with self.assertRaises(ValueError):solve_sparse_event([0,0,0,1],[-2,2,0,-1],1.5,cap=4.)

    def test_rigid_transform_preserves_feasibility_and_event(self):
        e=np.array([0.,0.,0.,1.]);s=np.array([-2.,2.,0.,-1.]);angle=.7
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        def transform(v):return np.r_[rotation@v[:2]+[100,-30],rotation@v[2:4]]
        a=solve_sparse_event(e,s,1.5);b=solve_sparse_event(transform(e),transform(s),1.5)
        self.assertTrue(b['feasible']);self.assertAlmostEqual(a['required_sparse_acceleration_norm_mps2'],b['required_sparse_acceleration_norm_mps2'])
        self.assertLess(b['sparse_event_residual_m'],1e-12)

    def test_invalid_future_or_nonfinite_state_rejected(self):
        for time in [0,3.1,float('nan')]:
            with self.assertRaises(ValueError):solve_sparse_event([0,0,0,1],[-2,2,0,-1],time)
        with self.assertRaises(ValueError):solve_sparse_event([0,0,0,float('nan')],[-2,2,0,-1],1.5)


if __name__=='__main__':unittest.main()
