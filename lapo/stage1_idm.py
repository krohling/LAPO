import wandb
import config
# import data_loader
import doy
import paths
import torch
import utils
import time
import random
from doy import loop

from flam.utils.plot.plot_comparison import plot_original_recon_rollout
from flam.loss.image import ImageLoss
from hdf5.hdf5_dataset import Hdf5Dataset
import hdf5.hdf5_data_loader as data_loader

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
train_data, valid_data, test_data = data_loader.load(
    **cfg.data,
    image_size=cfg.image_size,
    sub_traj_len=cfg.sub_traj_len,
)
train_iter = train_data.get_iter(cfg.stage1.bs)

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

    vq_loss, idm_stats = idm.label(batch)
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
        vq_loss=vq_loss,
        grad_norm=grad_norm,
        **idm_stats,
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


def test_multistep_prediction(dataset: Hdf5Dataset, n_steps: int=10, n_rec_episodes: int=10):
    idm.eval()  # disables idm.vq ema update
    wm.eval()

    # episodes = random.sample(test_data.dataset.get_all_episodes(), k=5)
    episodes = dataset.get_all_episodes()
    epi_rec_images_indices = random.sample(range(len(episodes)), k=min(n_rec_episodes, len(episodes)))
    rec_gt_images = {k: [] for k in epi_rec_images_indices}
    rec_pred_images = {k: [] for k in epi_rec_images_indices}

    with torch.no_grad():
        T = cfg.sub_traj_len
        losses = []
        counter = 0
        for epi_idx in range(len(episodes)):
            epi_steps = episodes[epi_idx]

            # randomly pick a safe rec_t to start recording for this episode
            rec_t = random.randint(0, max(0, epi_steps['image'].shape[0]-T-n_steps+1))

            for t in range(epi_steps['image'].shape[0]-T-n_steps+2):

                # Infer actions
                l_actions = []
                for i in range(n_steps):
                    idx = t+i
                    idm_x = epi_steps['image'][idx:idx+T].unsqueeze(0).to(config.DEVICE).contiguous()  # (1, T, C, H, W)
                    assert idm_x.shape[1] == T, f"idm_x.shape: {idm_x.shape}, T: {T}"

                    action_dict, _, _ = idm(idm_x)
                    l_actions.append(action_dict)

                # Predict images
                pred_xs = []
                for i in range(n_steps):
                    idx = t+i

                    # prepend real images if not enough predicted images
                    num_real_x = T - len(pred_xs) - 1
                    if num_real_x > 0:
                        x = epi_steps['image'][idx:idx+num_real_x].unsqueeze(0).to(config.DEVICE).contiguous()  # (1, num_real, C, H, W)
                        if len(pred_xs) > 0:
                            # (1, num_real, C, H, W) + (1, len(pred_xs), C, H, W) = (1, T-1, C, H, W)
                            x = torch.cat([x] + pred_xs, dim=1) 
                    else:
                        x = torch.cat(pred_xs, dim=1)  # (1, T-1, C, H, W)
                    
                    # infer next image
                    assert x.shape[1] == T-1, f"x.shape: {x.shape}, len(pred_xs): {len(pred_xs)}"
                    pred = wm(x, l_actions[i]['la']).clamp(0, 1)  # (1, C, H, W)
                    pred = pred.squeeze()

                    # compute loss using the last predicted image and target
                    target_idx = idx+T-1
                    target = epi_steps['image'][target_idx].to(config.DEVICE)
                    assert pred.shape == target.shape, f"pred.shape: {pred.shape}, target.shape: {target.shape}"
                    losses.append(image_loss(pred, target))

                    # from utils import save_seq_targ_pred_grids_labeled
                    # # print(f"x.shape: {x.shape}, target.shape: {target.shape}, pred.shape: {pred.shape}")
                    # save_seq_targ_pred_grids_labeled(
                    #     wm_in_seq=x.detach(),           # (B, T, C, H, W)
                    #     wm_targ=target.detach().unsqueeze(0),               # (B, C, H, W) or (B, T, C, H, W)
                    #     wm_pred=pred.detach().clamp(0, 1).unsqueeze(0),      # (B, C, H, W) or (B, T, C, H, W)
                    #     out_dir="rollout_vis",
                    #     prefix="step_{:07d}".format(counter),
                    #     num_samples=4,
                    #     repeat_target=True,
                    #     repeat_pred_if_single=True
                    # )
                    # counter += 1

                    if t == rec_t and epi_idx in epi_rec_images_indices:
                        if i == 0:
                            rec_gt_images[epi_idx].append(epi_steps['image'][target_idx-1][None, None].to(config.DEVICE))  # (1, 1, C, H, W)

                        rec_gt_images[epi_idx].append(target[None, None].to(config.DEVICE))  # (1, 1, C, H, W)
                        rec_pred_images[epi_idx].append(pred[None, None].to(config.DEVICE))  # (1, 1, C, H, W)

                    # append predicted image
                    if len(pred_xs) < T-1:
                        pred_xs.append(pred[None, None])
                    else:
                        pred_xs = pred_xs[1:] + [pred[None, None]] # fifo
            
            

        img_logging = {}
        for k in losses[0][1].keys():
            img_logging[k] = torch.stack([l[1][k] for l in losses]).mean()
        
        if len(epi_rec_images_indices) > 0:
            rec_gt_images = [torch.cat(rec_gt_images[k], dim=1) for k in epi_rec_images_indices]
            rec_pred_images = [torch.cat(rec_pred_images[k], dim=1) for k in epi_rec_images_indices]

            rec_images = plot_original_recon_rollout(
                images=torch.cat(rec_gt_images, dim=0), 
                images_rollout=torch.cat(rec_pred_images, dim=0)
            )

            rec_images = [wandb.Image(i, caption=f"Sample {i}") for i in rec_images]
            

            return img_logging, rec_images
        
        return img_logging, None


for step in loop(cfg.stage1.steps + 1, desc="[green bold](stage-1) Training IDM + FDM"):
    best_loss = float('inf')
    train_step()

    if step % cfg.stage1.eval_freq == 0:
        print("[bold green]Evaluating IDM + FDM on validation set...")
        start_time = time.time()
        val_losses, valid_image_samples = test_multistep_prediction(
            valid_data.dataset, 
            n_steps=cfg.stage1.n_eval_steps, 
            n_rec_episodes=cfg.stage1.n_valid_eval_sample_images
        )
        val_losses = {f"valid_{k}": v for k, v in val_losses.items()}
        logger(
            step, 
            global_step=step * cfg.stage1.bs, 
            valid_eval_duration=time.time()-start_time,
            valid_image_samples=valid_image_samples,
            **val_losses, 
        )

        if val_losses['valid_pixel_mse'] < best_loss:
            best_loss = val_losses['valid_pixel_mse']
            torch.save(
                dict(
                    **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                    step=step,
                    cfg=cfg,
                    logger=logger,
                ),
                paths.get_models_path(cfg.exp_name),
            )
            print(f"[bold green] New best model saved with pixel_mse: {best_loss:.6f}")
        else:
            # load best model
            print(f"[bold yellow] Validation pixel_mse did not improve from {best_loss:.6f}, loading best model")
            checkpoint = torch.load(paths.get_models_path(cfg.exp_name), map_location=config.DEVICE)
            idm.load_state_dict(checkpoint['idm'])
            wm.load_state_dict(checkpoint['wm'])
            opt.load_state_dict(checkpoint['opt'])
        
        print("[bold green]Evaluating IDM + FDM on test set...")
        start_time = time.time()
        test_losses, test_image_samples = test_multistep_prediction(
            test_data.dataset, 
            n_steps=cfg.stage1.n_eval_steps, 
            n_rec_episodes=cfg.stage1.n_test_eval_sample_images
        )
        test_losses = {f"test_{k}": v for k, v in test_losses.items()}
        logger(
            step, 
            global_step=step * cfg.stage1.bs, 
            test_eval_duration=time.time()-start_time,
            test_image_samples=test_image_samples,
            **test_losses, 
        )
        



    
