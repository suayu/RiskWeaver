from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm

from trajdata import AgentType, SceneBatch, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.visualization.vis import plot_scene_batch


def main():
    noise_hists = NoiseHistories()

    dataset = UnifiedDataset(
        desired_data=["nuplan_mini-mini_train"],
        centric="agent",
        desired_dt=0.1,
        history_sec=(1.0, 1.0),
        future_sec=(3.2, 3.2),
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30.0),
        incl_robot_future=True,
        incl_raster_map=True,
        raster_map_params={
            "px_per_m": 2,
            "map_size_px": 224,
            "offset_frac_xy": (-0.5, 0.0),
        },
        # augmentations=[noise_hists],
        max_agent_num=21,
        num_workers=4,
        verbose=True,
        data_dirs={  # Remember to change this to match your filesystem!
             "nuplan_mini":"/net/acadia3a/data/datasets/nuplan/dataset/nuplan-v1.1", 
        },
        check_cache=True,
        save_index=False
    )

    print(f"# Data Samples: {len(dataset):,}")

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        collate_fn=dataset.get_collate_fn(),
        num_workers=4,
    )

    batch: SceneBatch
    for batch in tqdm(dataloader):
        pass
        
        # plot_scene_batch(batch, batch_idx=0)


if __name__ == "__main__":
    main()
