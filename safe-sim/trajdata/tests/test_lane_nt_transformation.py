import unittest
import numpy as np
import unittest
import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader

from trajdata.data_structures.agent import AgentType
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.batch_element import AgentBatchElement, SceneBatchElement
from trajdata.data_structures.state import NP_STATE_TYPES, TORCH_STATE_TYPES
from trajdata.dataset import UnifiedDataset
from trajdata.custom_func.get_lane_info import get_lane_info

# Assuming you import the required functions from your actual code
# from get_lane_info import your_function_for_nt_transform, xy_from_nt_function
import torch
from trajdata.visualization.vis import plot_agent_batch

class TestGetLaneInfo(unittest.TestCase):
    def setUp(self):
        self.dataset = UnifiedDataset(
        desired_data=[
            # "nusc_trainval",
            # "nuplan_train",
            "nuplan_mini"
            # "nuplan_val",
            # "lyft_sample-mini_val",
        ],
        centric="agent",
        desired_dt=0.1,
        history_sec=(1.0, 1.0),
        future_sec=(2.0, 2.0),
        only_types=[AgentType.VEHICLE],
        # state_format="x,y,z,xd,yd,xdd,ydd,h",
        obs_format="x,y,z,xd,yd,xdd,ydd,s,c",
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=False,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        incl_vector_map=True,
        num_workers=0,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
            "nusc_trainval": "/net/acadia10a/data/sriram/nuscenes",
            "nuplan_train":"/net/acadia3a/data/datasets/nuplan/dataset/nuplan-v1.1",
            "nuplan_mini":"/net/acadia3a/data/datasets/nuplan/dataset/nuplan-v1.1",
            # "nuplan_val":"/net/acadia3a/data/datasets/nuplan/dataset/nuplan-v1.1/trainval",
            # "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
        },
        # A dictionary that contains functions that generate our custom data.
        # Can be any function and has access to the batch element.
        extras={
            "closest_lane_point": get_lane_info,
            # "all_possible_lane_pts": get_refs
        },
        # rebuild_cache=True,
        subset_fraction= 0.1,
        # cache_index=True
        )

        print(f"# Data Samples: {len(self.dataset):,}")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=self.dataset.get_collate_fn(pad_format="right"),
            num_workers=0,
        )
   
    def test_batch_operations(self):
        batch: AgentBatch
        for batch in self.dataloader:
            self._batch_assertions(batch)

    def _batch_assertions(self, batch):
        valid_batch_indices = np.where(batch.extras["has_lane"])[0]
        
        for batch_i in valid_batch_indices:
            agent_traj = batch.agent_hist[batch_i,:,:2]  # Replace this with the actual field
            centerline_xy = batch.extras["centerline_xy"][batch_i]
            
            assert_centerline_ahead(agent_traj, centerline_xy)
            if batch.extras["has_lane"][batch_i]:
                try:
                    assert_continuous_and_bounded(batch.extras["nt_arr_shift"][batch_i])
                    #reconstruction
                    gt_nt_arr = batch.extras["nt_delta_target"]
                    gt_nt_arr = torch.cumsum(gt_nt_arr, dim=1) + batch.extras["nt_current"].unsqueeze(1)
                    gt_xy_traj_batch = transform_FRT2Cart_poly_torch_batch(
                                    gt_nt_arr,
                                    batch.extras["polynomial_coeff"],
                                    batch.extras["t_boundaries"]
                                    )
                    agent_traj_fut = batch.agent_fut[batch_i,:,:2] 
                    self._assert_transformation_error(agent_traj_fut,gt_xy_traj_batch[batch_i])
                except AssertionError:
                    tested_path = torch.zeros_like(batch.extras["nt_delta_target"])
                    tested_path[:,:,1] = 1.0*5
                    tested_path[:,:,0] = 0.1
                    nt_arr = torch.cumsum(tested_path,dim=1)
                    xy_traj_batch = transform_FRT2Cart_poly_torch_batch(
                                                nt_arr,
                                                batch.extras["polynomial_coeff"],
                                                batch.extras["t_boundaries"]
                                                )
                    self._plot_scenario(batch, batch_i, gt_xy_traj_batch,xy_traj_batch)
            
    def _assert_transformation_error(self, xy_test_data, xy_reconstructed):
            error = np.linalg.norm(xy_reconstructed - xy_test_data)
            self.assertAlmostEqual(error, 0, places=2)

    def _assert_nt_smoothness(self, nt_arr):
        delta_nt_arr = np.diff(nt_arr, axis=0)
        n_threshold = 1.5
        t_threshold = 1.5
        n_violate = np.sum(np.abs(delta_nt_arr[:, 0]) > n_threshold)
        t_violate = np.sum(np.abs(delta_nt_arr[:, 1]) > t_threshold)
        self.assertEqual(n_violate, 0)
        self.assertEqual(t_violate, 0)
    def _plot_scenario(self, batch, batch_i, gt_xy_traj_batch,xy_traj_batch):
        import matplotlib.pyplot as plt
        ax = plot_agent_batch(
            batch, batch_idx=batch_i, legend=False, show=False, close=False
        )
        
        # Your existing plotting logic
        lane_points = batch.extras["all_poss_refs"][batch_i].reshape(-1, 2)
        lane_points = lane_points[
            torch.logical_not(torch.any(torch.isnan(lane_points), dim=1)), :
        ].numpy()
        ax.scatter(lane_points[:, 0], lane_points[:, 1], s=1, color="grey", label="All possible Lane points")
        # ... (snip)
        # Your existing plotting logic continues
        lane_points = batch.extras["all_poss_refs"][batch_i].reshape(-1,2)
        lane_points = lane_points[
            torch.logical_not(torch.any(torch.isnan(lane_points), dim=1)), :
        ].numpy()
      
        ax.scatter(
            lane_points[:, 0],
            lane_points[:, 1],
            # "o-",
            # markersize=3,
            s = 1,
            color = "grey",
            label="All possible Lane points",
            
        )

        closest_lane_points = batch.extras["centerline_xy"][batch_i]
        closest_lane_points = closest_lane_points[
            torch.logical_not(torch.any(torch.isnan(closest_lane_points), dim=1)), :
        ].numpy()

        ax.plot(
            closest_lane_points[:, 0],
            closest_lane_points[:, 1],
            "o-",
            markersize=1.5,
            color = "orange",
            label="Centerline points",
        )
        # ------------------------left right lane ---------------------------------------
        closest_lane_points = batch.extras["left_centerline_xy"][batch_i]
        closest_lane_points = closest_lane_points[
            torch.logical_not(torch.any(torch.isnan(closest_lane_points), dim=1)), :
        ].numpy()
        if not (closest_lane_points==0).all():
            ax.plot(
                closest_lane_points[:, 0],
                closest_lane_points[:, 1],
                "o-",
                markersize=1.5,
                color = "blue",
                label="left Centerline points",
            )

        closest_lane_points = batch.extras["right_centerline_xy"][batch_i]
        closest_lane_points = closest_lane_points[
            torch.logical_not(torch.any(torch.isnan(closest_lane_points), dim=1)), :
        ].numpy()
        if not (closest_lane_points==0).all():
            ax.plot(
                closest_lane_points[:, 0],
                closest_lane_points[:, 1],
                "o-",
                markersize=1.5,
                color = "red",
                label="right centerline points",
            )

        # -------------- fitted polynomials ---------------"
        # "

        ax.plot(
            xy_traj_batch[batch_i,:, 0],
            xy_traj_batch[batch_i,:, 1],
            "o-",
            markersize=1.5,
            color = "red",
            label="transfomed poly points",
        )

        ax.plot(
            gt_xy_traj_batch[batch_i,:, 0],
            gt_xy_traj_batch[batch_i,:, 1],
            "o-",
            markersize=1.5,
            color = "black",
            label="gt sd2xy points",
        )
        # Compute error for title
        error = torch.linalg.norm(batch.agent_fut[batch_i, ..., :2] - gt_xy_traj_batch[batch_i])
        
        ax.legend(loc="best", frameon=True)
        plt.title(str(error.item()))  # Assuming error is a tensor
        plt.show()
        plt.savefig(f"/net/ca-home1/home/mai/wjchang/trajdata/visualization/debug_lane_future_lanes{batch_i}_")
        plt.close("all")

def transform_FRT2Cart_poly_torch_batch(nt_arr: torch.Tensor, poly_coeffs: torch.Tensor, t_boundaries: torch.Tensor) -> torch.Tensor:
    """Transform 2D nt array from Frenet Serret coordinates to Cartesian space using piece-wise polynomials."""
    t_arr = nt_arr[:, :, 1:]
    n_arr = nt_arr[:, :, 0:1]

    xy_arr = torch.zeros_like(nt_arr)

    for i in range(t_boundaries.size(1)):
        if i == 0:
            segment_mask = nt_arr[:, :, 1] <= t_boundaries[:, i].unsqueeze(1)
        elif i == t_boundaries.size(1) - 1:
            segment_mask = nt_arr[:, :, 1] > t_boundaries[:, i - 1].unsqueeze(1)
        else:
            segment_mask = (nt_arr[:, :, 1] > t_boundaries[:, i - 1].unsqueeze(1)) & (nt_arr[:, :, 1] <= t_boundaries[:, i].unsqueeze(1))

        # subtract t_boundary from t for segments other than the first
        if i > 0:
            segment_t_arr = t_arr - t_boundaries[:, i - 1].unsqueeze(-1).unsqueeze(-1)
            assert (segment_t_arr[segment_mask]>=0).all() # make sure that after subtracting the t_boundary, the value is still >= 0
        else:
            segment_t_arr = t_arr

        segment_t_sqr_arr = torch.cat([segment_t_arr ** 4, segment_t_arr ** 3, segment_t_arr ** 2, segment_t_arr, torch.ones_like(segment_t_arr)], dim=-1)
        assert torch.isfinite(segment_t_sqr_arr).all(), segment_t_sqr_arr

        segment_poly_coeffs = poly_coeffs[:, i]

        # Apply mask to set values outside segment_mask to zero
        segment_t_sqr_arr[~segment_mask.unsqueeze(-1).expand_as(segment_t_sqr_arr)] = 0.0

        if segment_t_sqr_arr.sum() == 0:
            continue
        # Calculate the polynomial coefficients for x and y
        coeff_x, coeff_y = segment_poly_coeffs[:, 0], segment_poly_coeffs[:, 1]

        # Calculate the gradient of the polynomial coefficients
        grad_coeff_x = (coeff_x * torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0], device=coeff_x.device).unsqueeze(0))[:, :-1]
        grad_coeff_y = (coeff_y * torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0], device=coeff_y.device).unsqueeze(0))[:, :-1]

        # Calculate the tangent vector coefficients
        t_coeff = torch.stack((grad_coeff_x, grad_coeff_y), dim=1)

        # Calculate the normal vector coefficients
        normal_vec = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=t_coeff.device).repeat(t_coeff.shape[0],1,1)
        d_coeff = torch.bmm(normal_vec, t_coeff).transpose(1,2)

        # Offset the xy_arr by the normal vector coefficients
        d_arr = torch.bmm(segment_t_sqr_arr[..., 1:], d_coeff)

        # Normalize the d_arr if it is not zero
        d_arr_norm = torch.norm(d_arr, dim=-1, keepdim=True)
        d_arr_norm[d_arr_norm==0] = 1.0  # Set the norm to 1.0 for zero vectorss

        d_arr /= d_arr_norm

        xy_segment_arr = torch.bmm(segment_t_sqr_arr, segment_poly_coeffs.transpose(-1, -2))
        xy_segment_arr += n_arr * d_arr


        xy_arr[segment_mask.unsqueeze(-1).expand_as(xy_arr)]= xy_segment_arr[segment_mask.unsqueeze(-1).expand_as(xy_segment_arr)] #TODO check whether this is valid
        # xy_arr += xy_segment_arr
    return xy_arr
def assert_centerline_ahead(agent_traj_world, centerline_xy):
    """
    Asserts that the centerline is ahead of the agent.

    Parameters:
        agent_traj_world (numpy.ndarray): Agent's trajectory, shape (N, 4) where N is the number of points.
        centerline_xy (numpy.ndarray): Centerline points, shape (M, 2) where M is the number of points.
        
    Returns:
        None. Raises an AssertionError if the centerline is not ahead of the agent.
    """
    # The agent's current position
    agent_current_pos = agent_traj_world[0, :2]  # Only pick (x, y)
    
    # Compute the Euclidean distance from each centerline point to the agent's current position
    distances = np.linalg.norm(centerline_xy - agent_current_pos, axis=1)
    
    # Find the index of the closest point on the centerline
    closest_idx = np.argmin(distances)
    
    # Assert that the closest point should be ahead of the agent's position
    assert closest_idx > 0, "The centerline is not ahead of the agent."
def assert_continuous_and_bounded(nt_arr, max_n_rate = 10 *0.1 , max_t_rate = 30 *0.1):
    # Calculate rate of change for n and t
    delta_n = np.diff(nt_arr[:, 0])
    delta_t = np.diff(nt_arr[:, 1])

    #check the maximum rate between consecutive points in n and t directions
    assert np.all(delta_n <= max_n_rate), f"Discontinuity in n detected. Maximum observed rate: {nt_arr[:, 0]}"
    assert np.all(delta_t <= max_t_rate), f"Discontinuity in t detected. Maximum observed rate: {delta_t}"
    
if __name__ == '__main__':
    unittest.main()