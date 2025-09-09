import config
# import data_loader
from flam.loss.image import ImageLoss
import hdf5.hdf5_data_loader as data_loader
import doy
import paths
import torch
import utils
from doy import loop

cfg = config.get()
doy.print("[bold green]Running LAPO stage 1 (IDM/FDM training) with config:")
config.print_cfg(cfg)

cfg.sub_traj_len = max(cfg.sub_traj_len, 2)  # IDM needs at least 2 steps
config.set_add_time_horizon(cfg.sub_traj_len-2)

run, logger = config.wandb_init("lapo_stage1", config.get_wandb_cfg(cfg))

idm, wm = utils.create_dynamics_models(
    cfg.model, 
    image_size=cfg.image_size,
    sub_traj_len=cfg.sub_traj_len,
)


image_loss = ImageLoss(cfg.stage1.image_loss).to(config.DEVICE)
train_data, test_data = data_loader.load(
    **cfg.data,
    image_size=cfg.image_size,
    sub_traj_len=cfg.sub_traj_len,
)
train_iter = train_data.get_iter(cfg.stage1.bs)
# test_iter = test_data.get_iter(cfg.stage1.bs)

opt, lr_sched = doy.LRScheduler.make(
    all=(
        doy.PiecewiseLinearSchedule(
            [0, 50, cfg.stage1.steps + 1],
            [0.1 * cfg.stage1.lr, cfg.stage1.lr, 0.01 * cfg.stage1.lr],
        ),
        [wm, idm],
    ),
)


def train_step():
    idm.train()
    wm.train()

    lr_sched.step(step)

    batch = next(train_iter)

    vq_loss, vq_perp = idm.label(batch)
    wm_loss = wm.label(batch)
    loss = wm_loss + vq_loss

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([*idm.parameters(), *wm.parameters()], 2)
    opt.step()

    logger(
        step,
        wm_loss=wm_loss,
        global_step=step * cfg.stage1.bs,
        vq_perp=vq_perp,
        vq_loss=vq_loss,
        grad_norm=grad_norm,
        **lr_sched.get_state(),
    )


def test_step(eval_steps=10):
    idm.eval()  # disables idm.vq ema update
    wm.eval()

    # evaluate IDM + FDM generalization on (action-free) test data
    batch = next(test_iter)
    idm.label(batch)
    wm_loss = wm.label(batch)

    # train latent -> true action decoder and evaluate its predictiveness
    _, eval_metrics = utils.eval_latent_repr(train_data, idm)

    logger(step, wm_loss_test=wm_loss, global_step=step * cfg.stage1.bs, **eval_metrics)


def test_multistep_prediction():
    idm.eval()  # disables idm.vq ema update
    wm.eval()

    epi_steps = test_data.dataset.get_full_random_episode()

    with torch.no_grad():
        losses = []
        for i in range(0, epi_steps['image'].shape[0]-cfg.sub_traj_len):
            x = epi_steps['image'][i:i+cfg.sub_traj_len].unsqueeze(0).to(config.DEVICE).contiguous()  # (1, i, C, H, W)

            action_dict, _, _ = idm(x)
            pred = wm(x[:, :-1], action_dict['la'])
            
            pred = pred.squeeze()
            target = x[:, -1:].squeeze()
            assert pred.shape == target.shape, f"pred.shape: {pred.shape}, target.shape: {target.shape}"
            losses.append(image_loss(pred, target))

        episode_loss = torch.stack([l[0] for l in losses]).mean()
        img_logging = {}
        for k in losses[0][1].keys():
            img_logging[k] = torch.stack([l[1][k] for l in losses]).mean()
        
        # print("*"*20)
        # print(f"Step {step}: episode image loss: {episode_loss.item():.44f}")
        # for k in img_logging.keys():
        #     print(f"    {k}: {img_logging[k].item():.4f}")
        # print("*"*20)
        
        logger(step, img_loss=episode_loss, global_step=step * cfg.stage1.bs, **img_logging)


for step in loop(cfg.stage1.steps + 1, desc="[green bold](stage-1) Training IDM + FDM"):
    train_step()

    if step > 0 and step % cfg.stage1.eval_freq == 0:
        test_multistep_prediction()

    if step > 0 and (step % 5_000 == 0 or step == cfg.stage1.steps):
        torch.save(
            dict(
                **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                step=step,
                cfg=cfg,
                logger=logger,
            ),
            paths.get_models_path(cfg.exp_name),
        )
