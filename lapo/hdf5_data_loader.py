import random
from collections.abc import Generator
from pathlib import Path

import doy
import numpy as np
import torch
import torch.nn.functional as F
from config import ADD_TIME_HORIZON, DEVICE
from tensordict import TensorDict
from torch.utils.data import DataLoader
import tqdm

import h5py
from h5py import Group

from data_loader import  _unfold_td, normalize_obs, TRAIN_CHUNK_LEN, TEST_CHUNK_LEN

class HDF5DataStager:
    def __init__(
        self,
        hdf5_path: str,
        chunk_len: int,
        obs_depth: int = 3,
        seq_len: int = 2,
        image_dim: int = 64,
        obs_key: str = "obs"
    ) -> None:

        self.seq_len = seq_len
        self.chunk_len = chunk_len
        self.obs_depth = obs_depth
        self.image_dim = image_dim
        self.obs_key = obs_key

        self.episodes = []
        self.total_timesteps = 0
        self.td: TensorDict = None  # type: ignore
        self._unfold_td: TensorDict = None  # type: ignore

        self._load(hdf5_path)

    def _load(self, hdf5_path: str):
        with h5py.File(hdf5_path, "r") as hdf5_file:
            self.episodes = [e for e in list(hdf5_file.keys()) if isinstance(hdf5_file[e], Group)]
            random.shuffle(self.episodes)

            obs_tensors = []
            self.total_timesteps = sum(int(hdf5_file[ep].attrs["total_steps"]) for ep in self.episodes)

            for ep in tqdm.tqdm(self.episodes, desc="Loading episodes"):
                episode_length = int(hdf5_file[ep].attrs["total_steps"])
                obs_data = torch.tensor(np.array(hdf5_file[ep][self.obs_key][:episode_length]))
                
                if len(obs_data.shape) == 4:
                    if obs_data.shape[-1] == self.obs_depth:
                        # LAPO expects CHW format so we need to convert from HWC
                        # (T, H, W, C) -> (T, C, H, W)
                        obs_data = obs_data.permute(0, 3, 1, 2)

                    # Resize images to self.image_dim x self.image_dim if needed
                    if obs_data.shape[-1] != self.image_dim or obs_data.shape[-2] != self.image_dim:
                        obs_data = F.interpolate(
                            obs_data.float(),
                            size=(self.image_dim, self.image_dim),
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )

                        # Convert back to uint8. This has a big impact on RAM usage.
                        obs_data = obs_data.clamp_(0, 255).round_().to(torch.uint8)
            
                obs_tensors.append(obs_data)

            # Concat tensors and put them in the tensordict
            self.td = TensorDict(
                {
                    "obs": torch.cat(obs_tensors, dim=0).contiguous(),
                },
                batch_size=self.total_timesteps,
                device="cpu",
            )
            self.td_unfolded = _unfold_td(self.td, self.seq_len, 1)

    def get_iter(
        self,
        batch_size: int,
        device=DEVICE,
        shuffle=True,
        drop_last=True,
    ) -> Generator[TensorDict, None, None]:
        dataloader = DataLoader(
            self.td_unfolded,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=lambda x: x,
        )

        while True:
            for batch in dataloader:
                batch = batch.to(device)
                batch["obs"] = normalize_obs(batch["obs"])
                yield batch


def _load(hdf5_path: Path, test: bool, obs_key: str) -> HDF5DataStager:
    chunk_len = TEST_CHUNK_LEN if test else TRAIN_CHUNK_LEN
    return HDF5DataStager(
        hdf5_path=hdf5_path,
        chunk_len=chunk_len,
        seq_len=2 + ADD_TIME_HORIZON,
        obs_key=obs_key
    )


def load(env_name: str, hdf5_train_path: Path, hdf5_test_path: Path, obs_key: str="observations") -> tuple[HDF5DataStager, HDF5DataStager]:
    with doy.status(f"Loading expert data for train_path: {hdf5_train_path}"):
        dl_train = _load(hdf5_train_path, test=False, obs_key=obs_key)

    with doy.status(f"Loading expert data for test_path: {hdf5_test_path}"):
        ds_test = _load(hdf5_test_path, test=True, obs_key=obs_key)

    return dl_train, ds_test

