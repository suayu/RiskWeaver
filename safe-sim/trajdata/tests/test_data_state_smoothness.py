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
from trajdata.visualization.vis import plot_agent_batch,plot_agent_batch_state
from trajdata.custom_func.get_actions import get_actions_inverse_dynamics
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
            # "closest_lane_point": get_lane_info,
            "target_actions":get_actions_inverse_dynamics,
            # "all_possible_lane_pts": get_refs
        },
        # rebuild_cache=True,
        subset_fraction= 0.1,
        # cache_index=True
        )

        print(f"# Data Samples: {len(self.dataset):,}")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=self.dataset.get_collate_fn(pad_format="right"),
            num_workers=0,
        )
   
    def test_batch_operations(self):
        batch: AgentBatch
        for batch in self.dataloader:
            try:
                self._batch_assertions(batch)
            except AssertionError:
                plot_agent_batch_state(batch,batch_idx=0,name=f"{batch.data_idx[0]}_{batch.agent_name[0]}_max.png")
    def _batch_assertions(self, batch, max_acceleration=5.0, max_turning_rate=np.pi*3/4*0.1):
        agent_hist = batch.agent_hist
        
        # Extract velocity and acceleration
        velocity = agent_hist[..., 3:5]  # xd, yd
        acceleration = agent_hist[..., 5:7]  # xdd, ydd
        
        # Calculate orientation using arctan2
        orientation = torch.atan2(agent_hist[..., 8], agent_hist[..., 7])  # Assuming that s and c are sine and cosine
        turning_rate = torch.diff(orientation, dim=-1)

        # Check if any acceleration exceeds the max_acceleration
        accel_exceeds_max = (torch.norm(acceleration, dim=-1) > max_acceleration).any()

        # Check if any turning rate exceeds max_turning_rate
        turn_exceeds_max = (torch.abs(turning_rate) > max_turning_rate).any()

        self.assertTrue(not accel_exceeds_max and not turn_exceeds_max)

if __name__ == '__main__':
    unittest.main()
