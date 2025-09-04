from dataclasses import dataclass, field
import numpy as np
import torch
from typing import List, Union

@dataclass
class Hdf5DatasetConfig:
    """
    Assume the following directory structure:
        data_path
        ├── train_fname
        ├── valid_fname
        ├── test_fname
    """
    data_path: Union[str, List[str]] = "/scratch/cluster/zzwang_new/multigrid_data/tennis"

    train_fname: Union[str, List[str]] = "train.hdf5"
    valid_fname: Union[str, List[str]] = "valid.hdf5"
    test_fname: Union[str, List[str]] = "test.hdf5"

    frame_skip: Union[int, List[int]] = 4

    # if True, data will take the following timestamps (for frame_skip = 4)
    #   sub traj 1: [0, 4, 8, 12]
    #   sub traj 2: [1, 5, 9, 13]
    #   sub traj 3: [2, 6, 10, 14]
    #   sub traj 4: [3, 7, 11, 15]
    #   ...
    # otherwise, it will be (frames 1 - 3, 5 -7, ... will never be used)
    #   sub traj 1: [0, 4, 8, 12]
    #   sub traj 2: [4, 8, 12, 16]
    #   sub traj 3: [8, 12, 16, 20]
    #   ...
    iterate_frame_between_skip: bool | None = True

@dataclass
class DatasetConfig:
    dtype : torch.dtype = torch.float32
    np_dtype : np.dtype = np.float32
    rank: int = 0
    dataset: Hdf5DatasetConfig = field(default_factory=Hdf5DatasetConfig)