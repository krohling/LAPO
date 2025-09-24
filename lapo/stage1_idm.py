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
from einops import rearrange

from flam.utils.plot.plot_comparison import plot_original_recon_rollout
from flam.loss.image import ImageLoss
from hdf5.hdf5_dataset import Hdf5Dataset
import hdf5.hdf5_data_loader as data_loader
from flam.models.modules.tokenizer import Tokenizer

cfg = config.get()
doy.print("[bold green]Running LAPO stage 1 (IDM/FDM training) with config:")
config.print_cfg(cfg)

cfg.sub_traj_len = max(cfg.sub_traj_len, 2)  # IDM needs at least 2 steps
config.set_add_time_horizon(cfg.sub_traj_len-2)

run, logger = config.wandb_init("lapo_stage1", config.get_wandb_cfg(cfg))

tokenizer = Tokenizer(
    cfg.model.tokenizer_load_path,
    sub_traj_len=cfg.sub_traj_len,
).to(config.DEVICE).eval()

idm, wm = utils.create_dynamics_models(
    cfg.model, 
    feat_shape=cfg.feat_shape,
    sub_traj_len=cfg.sub_traj_len,
)


image_loss = ImageLoss(cfg.stage1.image_loss).eval().to(config.DEVICE)
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

if cfg.stage1.load_checkpoint is not None and cfg.stage1.load_checkpoint != "":
    print(f"[bold yellow] Loading checkpoint from {cfg.stage1.load_checkpoint} ...")
    checkpoint = torch.load(cfg.stage1.load_checkpoint, map_location=config.DEVICE, weights_only=False)
    idm.load_state_dict(checkpoint['idm'])
    wm.load_state_dict(checkpoint['wm'])
    opt.load_state_dict(checkpoint['opt'])
    print(f"[bold yellow] Loaded checkpoint from {cfg.stage1.load_checkpoint}.")

def tokenize_input(x):
    # Encoder input: (B, T, C, H, W) -> (B, T, d_feat, h_feat, w_feat)
    with torch.no_grad():
        # print(f"Tokenizing input with shape {x.shape}")
        x = tokenizer.encode(x)[0]
        x = rearrange(x, 'b t h w d -> b t d h w').contiguous()
        # print(f"Tokenized input to shape {x.shape}")
        
    return x

# decode predicted features back to images for logging
def decode_to_image(x):
    # Decoder input: (B, d_feat, h_feat, w_feat) -> (B, C, H, W)
    with torch.no_grad():
        x = rearrange(x, 'b d h w -> b h w d').contiguous()
        x = tokenizer.decode(x)
        
    return x

def train_step():
    idm.train()
    wm.train()

    lr_sched.step(step)

    batch = next(train_iter)
    batch['obs'] = tokenize_input(batch['obs'])
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


def test_multistep_prediction(
        dataset: Hdf5Dataset, 
        n_steps: int=10, 
        n_rec_episodes: int=5, 
        use_ds_percentage: float=1.0,
        print_progress: bool=False,
    ):
    idm.eval()  # disables idm.vq ema update
    wm.eval()


    episodes = dataset.get_all_episodes()
    if use_ds_percentage < 1.0:
        total_episodes = len(episodes)
        n_episodes = int(total_episodes * use_ds_percentage)
        episodes = random.sample(episodes, k=n_episodes)
        print(f"Using {n_episodes} episodes ({use_ds_percentage*100:.1f}%) for evaluation, out of {total_episodes} total episodes.")

    epi_rec_images_indices = random.sample(range(len(episodes)), k=min(n_rec_episodes, len(episodes)))
    rec_gt_images = {k: [] for k in epi_rec_images_indices}
    rec_pred_images = {k: [] for k in epi_rec_images_indices}

    with torch.no_grad():
        T = cfg.sub_traj_len
        pred_losses, rand_pred_losses = [], []
        counter = 0
        for epi_idx in range(len(episodes)):
            if print_progress:
                print(f"Evaluating episode {epi_idx+1} of {len(episodes)}")
            epi_steps = episodes[epi_idx]

            # randomly pick a safe rec_t to start recording for this episode
            rec_t = random.randint(0, max(0, epi_steps['image'].shape[0]-T-n_steps+1))

            for t in range(epi_steps['image'].shape[0]-T-n_steps+2):
                idm_xs = torch.cat([
                    epi_steps['image'][t+i:t+i+T].unsqueeze(0).to(config.DEVICE).contiguous() 
                    for i in range(n_steps)
                ])  # (n_steps, T, C, H, W)
                assert idm_xs.shape[1] == T, f"idm_xs.shape: {idm_xs.shape}, T: {T}"

                idm_xs = tokenize_input(idm_xs)  # (n_steps, T, d_feat, h_feat, w_feat)
                l_actions, _, _ = idm(idm_xs)
                l_rand_actions = idm.sample_la(n_steps)  # (n_steps, latent_action_dim)

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
                    
                    # infer next image using both the predicted and random latent actions
                    assert x.shape[1] == T-1, f"x.shape: {x.shape}, len(pred_xs): {len(pred_xs)}"
                    wm_xs = tokenize_input(
                        torch.cat([x, x], dim=0)  # (2, T-1, d_feat, h_feat, w_feat)
                    )
                    wm_out = wm(
                        wm_xs, 
                        torch.cat([
                            l_actions['la'][i].unsqueeze(0), 
                            l_rand_actions[i].unsqueeze(0)
                        ], dim=0) # (2, latent_action_dim) 
                    ).clamp(0, 1)  # (2, d_feat, h_feat, w_feat) [pred, rand_pred]
                    wm_out = decode_to_image(wm_out)  # (2, C, H, W)
                    pred, rand_pred = wm_out[0].squeeze(), wm_out[1].squeeze()  # (C, H, W), (C, H, W)

                    # compute loss using the last predicted image and target
                    target_idx = idx+T-1
                    target = epi_steps['image'][target_idx].to(config.DEVICE)
                    assert pred.shape == target.shape, f"pred.shape: {pred.shape}, target.shape: {target.shape}"
                    pred_losses.append(image_loss(pred, target)[1])
                    assert rand_pred.shape == target.shape, f"rand_pred.shape: {rand_pred.shape}, target.shape: {target.shape}"
                    rand_pred_losses.append(image_loss(rand_pred, target)[1])

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
                            rec_gt_images[epi_idx].append(epi_steps['image'][target_idx-1][None, None].detach().cpu())  # (1, 1, C, H, W)

                        rec_gt_images[epi_idx].append(target[None, None].detach().cpu())  # (1, 1, C, H, W)
                        rec_pred_images[epi_idx].append(pred[None, None].detach().cpu())  # (1, 1, C, H, W)

                    # append predicted image
                    if len(pred_xs) < T-1:
                        pred_xs.append(pred[None, None])
                    else:
                        pred_xs = pred_xs[1:] + [pred[None, None]] # fifo
                    
                    torch.cuda.empty_cache()
            
            

        img_logging = {}
        for k in pred_losses[0].keys():
            img_logging[k] = torch.stack([l[k] for l in pred_losses]).mean()
            img_logging[f"rand_{k}"] = torch.stack([l[k] for l in rand_pred_losses]).mean()
            img_logging[f"diff_{k}"] = img_logging[k] - img_logging[f"rand_{k}"]
        
        if len(epi_rec_images_indices) > 0:
            rec_gt_images = [torch.cat(rec_gt_images[k], dim=1) for k in epi_rec_images_indices if len(rec_gt_images[k]) > 0]
            rec_pred_images = [torch.cat(rec_pred_images[k], dim=1) for k in epi_rec_images_indices if len(rec_pred_images[k]) > 0]

            rec_images = plot_original_recon_rollout(
                images=torch.cat(rec_gt_images, dim=0), 
                images_rollout=torch.cat(rec_pred_images, dim=0),
                num_images=len(rec_gt_images)
            )

            wandb_imgs = [wandb.Image(i, caption=f"Sample {i}") for i in rec_images]
            
            return img_logging, wandb_imgs
        
        return img_logging, None


best_loss = float('inf')
best_checkpoint_path = None
for step in loop(cfg.stage1.steps + 1, desc="[green bold](stage-1) Training IDM + FDM"):
    if cfg.stage1.only_eval_test:
        break
    
    train_step()

    if step > cfg.stage1.eval_skip_steps and step % cfg.stage1.eval_freq == 0:
        current_checkpoint_path = paths.get_experiment_dir(cfg.exp_name) / f"checkpoint_{step:07d}.pt"
        torch.save(
            dict(
                **doy.get_state_dicts(wm=wm, idm=idm, opt=opt),
                step=step,
                cfg=cfg,
            ),
            current_checkpoint_path,
        )

        print("[bold green]Evaluating IDM + FDM on validation set...")
        start_time = time.time()
        val_losses, valid_image_samples = test_multistep_prediction(
            valid_data.dataset, 
            n_steps=cfg.stage1.n_eval_steps, 
            n_rec_episodes=cfg.stage1.n_valid_eval_sample_images,
            use_ds_percentage=cfg.stage1.valid_dataset_percentage,
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
            best_checkpoint_path = current_checkpoint_path
            print(f"[bold yellow] New best model with validation loss={best_loss:.6f} saved to {best_checkpoint_path}.")


if best_checkpoint_path is not None:
    print(f"[bold yellow] Loading best checkpoint with validation loss={best_loss:.6f}, from {best_checkpoint_path} ...")
    checkpoint = torch.load(best_checkpoint_path, map_location=config.DEVICE, weights_only=False)
    idm.load_state_dict(checkpoint['idm'])
    wm.load_state_dict(checkpoint['wm'])
    opt.load_state_dict(checkpoint['opt'])

print("[bold green]Evaluating best checkpoint on test set...")
start_time = time.time()
test_losses, test_image_samples = test_multistep_prediction(
    test_data.dataset, 
    n_steps=cfg.stage1.n_eval_steps, 
    n_rec_episodes=cfg.stage1.n_test_eval_sample_images,
    use_ds_percentage=1.0,
    print_progress=True,
)
test_losses = {f"test_{k}": v for k, v in test_losses.items()}
logger(
    step, 
    global_step=step * cfg.stage1.bs, 
    test_eval_duration=time.time()-start_time,
    test_image_samples=test_image_samples,
    **test_losses, 
)

print(test_losses)

print("[bold green]Finished LAPO stage 1 (IDM/FDM training).")
