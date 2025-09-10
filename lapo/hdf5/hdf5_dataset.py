import dataclasses
from typing import List, Dict, Literal, Tuple, Union, Optional

import h5py

from tqdm import tqdm
from pathlib import Path

import numpy as np
from einops import rearrange

import torch
import torchvision.transforms as T

from hdf5.hdf5_cfg import Hdf5DatasetConfig

@dataclasses.dataclass
class SubTrajectory:
    image: torch.FloatTensor                                                # (bs, T, C, H, W)
    action: Optional[Union[torch.LongTensor, torch.FloatTensor]] = None     # (bs, T)
    reward: Optional[torch.FloatTensor] = None                              # (bs, T)
    termination: Optional[torch.BoolTensor] = None                          # (bs, T)
    truncation: Optional[torch.BoolTensor] = None                           # (bs, T)

    def pin_memory(self):
        self.image = self.image.pin_memory()
        if self.action is not None:
            self.action = self.action.pin_memory()
        if self.reward is not None:
            self.reward = self.reward.pin_memory()
        if self.termination is not None:
            self.termination = self.termination.pin_memory()
        if self.truncation is not None:
            self.truncation = self.truncation.pin_memory()
        return self

class Hdf5Dataset(torch.utils.data.Dataset):
    """
    Assume the following directory structure:
        1. if data_path is str and train_fname/valid_fname/test_fname is str:
            data_path
            ├── train_fname
            ├── valid_fname
            ├── test_fname
        2. if data_path is List[str]:
            data_path[0]
            ├── train_fname
            ├── valid_fname
            ├── test_fname
            data_path[1]
            ├── train_fname
            ├── valid_fname
            ├── test_fname
            ...
        3. if train_fname/valid_fname/test_fname is List[str] or str with wildcard (*):
            data_path
            ├── train_fname[0]
            ├── train_fname[1]
            ├── ...
            ├── valid_fname[0]
            ├── valid_fname[1]
            ├── ...
            ├── test_fname[0]
            ├── test_fname[1]
            ├── ...
    Assume the each file is a hdf5 file with the following structure:
        file
        ├── attrs["num_videos"]: # of episodes in the file (int)
        ├── episode_0
        |   ├── attrs["total_steps"]: # of frames in the episode (int)
        |   └── observations (T, H, W, C), np.uint8
        ...
    """

    def __init__(
        self,
        config: Hdf5DatasetConfig,
        split: Literal["train", "valid", "test"],
        sub_traj_len: int,
        image_size: Union[int, Tuple[int, int]],
    ):
        self.config = config

        self.dtype = config.dtype
        self.np_dtype = config.np_dtype

        self.split = split
        self.sub_traj_len = sub_traj_len
        self.image_size = image_size
        self.resize = T.Resize((self.image_size, self.image_size))

        # === Normalize configuration for paths, file names, and frame_skip ===
        data_path = config.dataset.data_path
        split_fname = {
            "train": config.dataset.train_fname,
            "valid": config.dataset.valid_fname,
            "test": config.dataset.test_fname,
        }[split]

        if isinstance(data_path, str):
            data_path = [data_path]
        if isinstance(split_fname, str):
            split_fname = [split_fname]

        data_paths = []
        for data_path_i in data_path:
            for split_fname_i in split_fname:
                base = Path(data_path_i)
                candidate = base / split_fname_i

                # split_fname_i may contain wildcard, expand wildcard if so
                if any(ch in split_fname_i for ch in ["*", "?", "["]):
                    pattern = str(candidate)
                    matches = sorted({Path(p) for p in __import__('glob').glob(pattern)})
                    assert len(matches) > 0, f"No files matched pattern: {pattern}"
                    data_paths.extend(matches)
                else:
                    data_paths.append(candidate)

        # Validate data paths exist
        for data_path in data_paths:
            assert data_path.exists(), f"Data path {data_path} does not exist"

        frame_skips = config.dataset.frame_skip
        if isinstance(frame_skips, int):
            frame_skips = [frame_skips] * len(data_paths)
        else:
            assert len(frame_skips) == len(data_paths), "frame_skip list must align with number of files"

        # === Build Mapping from Index to Data ===
        self.sub_traj_paths = []
        total_num_episodes = 0
        num_timestamps = 0
        for data_idx, (data_path, frame_skip) in enumerate(zip(data_paths, frame_skips)):
            with h5py.File(data_path, "r") as f:
                if "num_videos" in f.attrs:
                    num_episodes = int(f.attrs["num_videos"])  # type: ignore[index]
                else:
                    num_episodes = len([
                        e for e in list(f.keys()) 
                        if "episode" in e and isinstance(f[e], h5py.Group)
                    ])
                episode_indices = sorted([
                    int(ep.split('_')[1]) for ep in f.keys() 
                    if "episode_" in ep and isinstance(f[ep], h5py.Group) and "observations" in f[ep]
                ])

            epi_idx_to_steps = self._get_episode_steps(str(data_path), episode_indices)

            for epi_idx in tqdm(
                episode_indices,
                disable=config.rank != 0,
                desc=f"Loading {split} dataset ({data_idx + 1}/{len(data_paths)})",
            ):
                total_steps = int(epi_idx_to_steps[epi_idx])

                if total_steps <= 0:
                    continue

                num_timestamps += total_steps

                if config.dataset.iterate_frame_between_skip:
                    effective_len = (self.sub_traj_len - 1) * frame_skip + 1
                    for timestamp in range(max(0, total_steps - effective_len + 1)):
                        self.sub_traj_paths.append((data_path, frame_skip, epi_idx, timestamp))
                else:
                    for timestamp in range(total_steps // frame_skip + 1 - (self.sub_traj_len - 1)):
                        self.sub_traj_paths.append((data_path, frame_skip, epi_idx, timestamp * frame_skip))

            total_num_episodes += num_episodes

        print(f"{split} dataset has: {total_num_episodes} episodes, {num_timestamps} timestamps, and {len(self.sub_traj_paths)} data points")

    @property
    def action_space(self):
        return None

    def __len__(self):
        return len(self.sub_traj_paths)

    @staticmethod
    def _get_episode_steps(data_file_path: str, episode_indices: List[int]):
        """
        Read per-episode total_steps from a single cache dataset "episode_lengths"
        if available; otherwise compute once and write it to the HDF5 file for
        future runs. The cache is an array aligned with the provided episode_indices.
        Returns a mapping from episode index to total_steps.
        """
        try:
            with h5py.File(data_file_path, "r", libver="latest", swmr=True) as f:
                if "episode_lengths" in f:
                    episode_lengths = f["episode_lengths"][...].astype(np.int64).tolist()
                    if len(episode_lengths) == len(episode_indices):
                        return {int(idx): int(steps) for idx, steps in zip(episode_indices, episode_lengths)}
        except Exception:
            pass

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Compute for all episodes and write the consolidated cache
        episode_lengths = []
        with h5py.File(data_file_path, "r", libver="latest", swmr=True) as f:
            for epi_idx in episode_indices:
                epi_group = f[f"episode_{epi_idx}"]
                assert isinstance(epi_group, h5py.Group)
                total_steps = int(epi_group.attrs["total_steps"])  # type: ignore[index]
                episode_lengths.append(total_steps)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            try:
                with h5py.File(data_file_path, "a") as f:
                    # Overwrite or create the single cache dataset
                    if "episode_lengths" in f:
                        del f["episode_lengths"]
                    f.create_dataset("episode_lengths", data=np.array(episode_lengths, dtype=np.int64))
            except Exception:
                print(f"Warning: could not write episode_lengths to {data_file_path}")

        return {int(idx): int(steps) for idx, steps in zip(episode_indices, episode_lengths)}

    def get_full_random_episode(self) -> Dict[str, torch.Tensor]:
        rand_idx = np.random.randint(len(self.sub_traj_paths))
        data_path, _, epi_idx, _ = self.sub_traj_paths[rand_idx]
        with h5py.File(data_path, "r", libver="latest", swmr=True) as f:
            epi_group = f[f"episode_{epi_idx}"]
            assert isinstance(epi_group, h5py.Group)
            total_steps = int(epi_group.attrs["total_steps"])  # type: ignore[index]
            image = np.ascontiguousarray(epi_group["observations"][0:total_steps])              # (T, H, W, C)
        
        image = (torch.from_numpy(image).to(self.dtype) / 255.0) # - 0.5                  # (T, H, W, C)
        image = rearrange(image, "t h w c -> t c h w")                                  # (T, C, H, W)
        image = self.resize(image)                                                      # (T, C, H', W')
        image = image.clip(0.0, 1.0)
        return {
            "image": image.to(dtype=self.dtype),
        }
        

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        data_path, frame_skip, epi_idx, timestamp = self.sub_traj_paths[idx]
        with h5py.File(data_path, "r", libver="latest", swmr=True) as f:
            epi_group = f[f"episode_{epi_idx}"]
            assert isinstance(epi_group, h5py.Group)
            effective_len = (self.sub_traj_len - 1) * frame_skip + 1
            idxes = slice(timestamp, timestamp + effective_len, frame_skip)

            image = np.ascontiguousarray(epi_group["observations"][idxes])              # (T, H, W, C)

        image = (torch.from_numpy(image).to(self.dtype) / 255.0) # - 0.5                  # (T, H, W, C)
        image = rearrange(image, "t h w c -> t c h w")                                  # (T, C, H, W)
        image = self.resize(image)                                                      # (T, C, H', W')
        image = image.clip(0.0, 1.0)

        sub_traj = {
            "image": image.to(dtype=self.dtype),
        }

        return sub_traj

    @staticmethod
    def collate_fn(batch_list: List[Dict[str, torch.Tensor]]):
        batch = torch.utils.data.default_collate(batch_list)
        batch = SubTrajectory(**batch)
        return batch
