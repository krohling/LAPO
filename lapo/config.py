from dataclasses import dataclass, field
from typing import Iterable, Literal

import doy
import env_utils
import rich
from omegaconf import DictConfig, OmegaConf
from rich.syntax import Syntax
import torch
import wandb

from flam.configs.loss.image_cfg import ImageLossConfig
from flam.configs.models.modules.encoder import EncoderConfig
from flam.configs.models.modules.decoder import DecoderConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

""" 
The actual config is in config.yaml, these dataclasses are just for validation and type hints.
"""

add_time_horizon: int = 0

def get_add_time_horizon() -> int:
    return add_time_horizon

def set_add_time_horizon(v: int) -> None:
    global add_time_horizon
    add_time_horizon = v

@dataclass
class VQConfig:
    enabled: bool
    num_codebooks: int
    num_discrete_latents: int
    emb_dim: int
    num_embs: int
    commitment_cost: float
    decay: float

@dataclass
class WMEncDecConfig:
    encoder_type: str = "impala"  # Literal["magvit2", "impala", "cosmos_continuous"]
    decoder_type: str = "lapo"    # Literal["magvit2", "lapo", "cosmos_continuous"]
    encoder_all: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_all: DecoderConfig = field(default_factory=DecoderConfig)

@dataclass
class IDMEncoderConfig:
    encoder_type: str = "impala"    # Literal["default", "magvit2", "impala", "cosmos_continuous"]
    encoder_all: EncoderConfig = field(default_factory=EncoderConfig)


@dataclass
class ModelConfig:
    wm_scale: int
    policy_impala_scale: int
    vq: VQConfig
    la_dim: int
    ta_dim: int
    idm_encoder: IDMEncoderConfig = field(default_factory=IDMEncoderConfig)
    wm_encdec: WMEncDecConfig = field(default_factory=WMEncDecConfig)

@dataclass
class DataConfig:
    path: str
    train_fname: str = "train.hdf5"
    test_fname: str = "test.hdf5"
    frame_skip: int = 4
    iterate_frame_between_skip: bool | None = True
    num_workers: int = 0
    prefetch_factor: int | None = None


@dataclass
class Stage1Config:
    """Hyperparams for training the latent IDM (+FDM)"""

    lr: float
    bs: int
    steps: int
    image_loss: ImageLossConfig = field(default_factory=ImageLossConfig)


@dataclass
class Stage2Config:
    """Hyperparams for behavior cloning the latent policy (from the latent IDM)"""

    lr: float
    bs: int
    steps: int


@dataclass
class Stage3Config:
    """PPO hyperparams for stage 3 (decoding a latent policy online)"""

    steps: int
    num_envs: int
    grad_accum_f: int
    num_steps: int
    num_minibatches: int
    update_epochs: int
    ent_coef: float
    lr: float
    anneal_lr: bool
    norm_adv: bool
    clip_coef: float
    clip_vloss: bool
    vf_coef: float
    max_grad_norm: float
    target_kl: float | None
    gamma: float
    gae_lambda: float
    batch_size: int
    minibatch_size: int


@dataclass
class Config:
    env_name: str
    exp_name: str
    stage_exp_name: str | None

    data: DataConfig
    model: ModelConfig

    stage1: Stage1Config
    stage2: Stage2Config
    stage3: Stage3Config

    image_size: int = 64
    sub_traj_len: int = 2


def get(
    base_cfg: DictConfig | None = None,
    use_cli_args: bool = True,
    override_args: list[str] | None = None,
    reload_keys: tuple[str, ...] = (),
) -> Config:
    """Initialize a config (either from config.yaml or from the base_cfg), apply cli and override flags, and validate it."""

    file_cfg = OmegaConf.load("config.yaml")
    if base_cfg is not None:
        # base_cfg is a structured config and won't let us update it with a generic dict
        # Instead we turn base_cfg into a DictConfig, then update it with reloaded keys,
        # and then apply other patches and checks
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg))
        cfg.update(OmegaConf.masked_copy(file_cfg, reload_keys))  # type: ignore
    else:
        cfg = file_cfg

    # apply any overrides
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    if override_args is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_args))

    _apply_runtime_vals(cfg)  # type: ignore

    cfg_schema: Config = OmegaConf.structured(Config)
    return OmegaConf.merge(cfg_schema, cfg)


def _apply_runtime_vals(cfg: Config) -> None:
    """Compute runtime values and add them to config"""

    # compute true and latent action dimensions
    cfg.model.la_dim = (
        cfg.model.vq.num_codebooks
        * cfg.model.vq.num_discrete_latents
        * cfg.model.vq.emb_dim
    )
    cfg.model.ta_dim = env_utils.ta_dim[cfg.env_name]

    cfg.stage3.batch_size = int(cfg.stage3.num_envs * cfg.stage3.num_steps)
    cfg.stage3.minibatch_size = int(cfg.stage3.batch_size // cfg.stage3.num_minibatches)


def print_cfg(cfg: Config, exclude_keys: Iterable[str] = ()):
    cfg = cfg.copy()  # type: ignore
    for k in exclude_keys:
        delattr(cfg, k)
    printer = rich if doy.progress._rich_console is None else doy.progress._rich_console
    printer.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def get_wandb_cfg(cfg: Config) -> dict:
    """transform config to dict for wandb logging and add other metadata"""
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    if cfg.env_name in env_utils.procgen_names:
        cfg_dict["env_key"] = (
            f"{env_utils.procgen_names.index(cfg.env_name)}-{cfg.env_name}"
        )
    else:
        cfg_dict["env_key"] = cfg.env_name
    return cfg_dict


def wandb_init(
    project: str,
    config: dict,
    wandb_enabled: bool = True,
    wandb_tags: list[str] | None = None,
):
    run_name = config["exp_name"]
    if config["stage_exp_name"]:
        run_name += f"-{config['stage_exp_name']}"

    run = wandb.init(
        project=project,
        config=config,
        mode="online" if wandb_enabled else "disabled",
        tags=wandb_tags,
        save_code=True,
        name=run_name,
        id=run_name,
        anonymous="allow",
    )

    return run, doy.Logger(use_wandb=wandb_enabled)