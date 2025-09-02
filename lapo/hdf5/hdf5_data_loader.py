from collections.abc import Generator
from typing import List, Union

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from config import DEVICE
from hdf5.hdf5_cfg import DatasetConfig, Hdf5DatasetConfig
from hdf5.hdf5_dataset import Hdf5Dataset

class HDF5DataStager:
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        sub_traj_len: int = 2,
        image_size: int = 64,
        train_fname: str = "train.hdf5",
        valid_fname: str = "valid.hdf5",
        test_fname: str = "test.hdf5",
        frame_skip: Union[int, List[int]] = 4,
        iterate_frame_between_skip: bool | None = True,
        dtype=torch.float32,
        np_dtype=np.float32,
        **kwargs
    ) -> None:
        self.dataset = Hdf5Dataset(
            config=DatasetConfig(
                dtype=dtype,
                np_dtype=np_dtype,
                rank=0,
                dataset=Hdf5DatasetConfig(
                    data_path=data_path,
                    train_fname=train_fname,
                    valid_fname=valid_fname,
                    test_fname=test_fname,
                    frame_skip=frame_skip,
                    iterate_frame_between_skip=iterate_frame_between_skip,
                )
            ),
            split=split,
            sub_traj_len=sub_traj_len,
            image_size=image_size,
        )
    
    def get_iter(
        self,
        batch_size: int,
        device=DEVICE,
        shuffle=True,
        drop_last=True,
    ) -> Generator[TensorDict, None, None]:
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: x,
        )

        while True:
            for batch in dataloader:
                obs_tensor = torch.stack([i["image"] for i in batch], dim=0).contiguous().to(device)
                print(f"obs_tensor shape: {obs_tensor.shape}  min(): {obs_tensor.min()}, max: {obs_tensor.max()}, dtype: {obs_tensor.dtype}")

                b_dict = TensorDict(
                    {
                        "obs": obs_tensor,
                    },
                    batch_size=len(batch),
                    device=device,
                )

                yield b_dict


def load(
        data_path: str, 
        **kwargs
    ) -> tuple[HDF5DataStager, HDF5DataStager]:
    
    dl_train = HDF5DataStager(
        data_path=data_path,
        split="train",
        **kwargs
    )

    ds_test = HDF5DataStager(
        data_path=data_path,
        split="test",
        **kwargs
    )

    return dl_train, ds_test

