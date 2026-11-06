import os

from trajdata import UnifiedDataset


# def main():
#     dataset = UnifiedDataset(
#         desired_data=["nusc_mini", "lyft_sample", "nuplan_mini"],
#         rebuild_cache=True,
#         rebuild_maps=True,
#         num_workers=os.cpu_count(),
#         verbose=True,
#         data_dirs={  # Remember to change this to match your filesystem!
#             "nusc_mini": "~/datasets/nuScenes",
#             "lyft_sample": "~/datasets/lyft/scenes/sample.zarr",
#             "nuplan_mini": "~/datasets/nuplan/dataset/nuplan-v1.1",
#         },
#     )
#     print(f"Total Data Samples: {len(dataset):,}")
def main():
    dataset = UnifiedDataset(
        desired_data=["nusc_mini", "nusc_test", "nusc_trainval"],  # 只处理 nuscenes 数据集
        rebuild_cache=True,
        rebuild_maps=True,
        num_workers=os.cpu_count(),
        verbose=True,
        data_dirs={  # 确保路径正确
            # "nusc_mini": "/hqlab/dataset_nas1/nuscenes/v1.0-mini",
            # "nusc_trainval": "/hqlab/dataset_nas1/nuscenes/v1.0-trainval",
            # "nusc_test": "/hqlab/dataset_nas1/nuscenes/v1.0-test",
            "nusc_mini": "/hqlab/dataset_nas1/nuscenes",
            "nusc_trainval": "/hqlab/dataset_nas1/nuscenes",
            "nusc_test": "/hqlab/dataset_nas1/nuscenes",
        },
    )
    print(f"Total Data Samples: {len(dataset):,}")


if __name__ == "__main__":
    main()