"""
PickMoment training on GoPro.

Model: f_theta(x_blur, s, t) -> mean image over the exposure sub-interval [s, t]

Losses (weights in config `loss.*`):
    recon   : f(s, t) vs mean of GT sub-frames in [s, t]          (latent L2 by default)
    ident   : f(t - eps, t + eps) vs sharp frame at t             (pixel L1 + edge-aware LPIPS)
    consist : additivity (t-s) f(s,t) = (m-s) f(s,m) + (t-m) f(m,t)  (latent L2, RHS detached)

Usage:
    accelerate launch train.py --config configs/train.yaml
"""

import argparse
import logging
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models import PickMoment, PickMomentArgs
from dataloader import GoProTrainDataset, GoProValDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# Datasets / dataloaders (progressive patch schedule)
# --------------------------------------------------------------------------

def get_progressive_config(global_step, config):
    """(patch_size, batch_size) for the current step."""
    data_cfg = config['data']
    progressive = data_cfg.get('progressive', {})
    schedule = progressive.get('schedule', [])
    if not progressive.get('enabled', False) or not schedule:
        return data_cfg['patch_size'], data_cfg['batch_size']

    for patch_size, batch_size, until_step in schedule:
        if global_step < until_step:
            return patch_size, batch_size
    last = schedule[-1]
    return last[0], last[1]


def get_stage_for_step(global_step, config):
    progressive = config['data'].get('progressive', {})
    if not progressive.get('enabled', False):
        return 0
    schedule = progressive.get('schedule', [])
    for i, (_, _, until_step) in enumerate(schedule):
        if global_step < until_step:
            return i
    return len(schedule) - 1


def create_train_dataset(config, patch_size):
    data_cfg = config['data']
    return GoProTrainDataset(
        gopro_dir=data_cfg['gopro_train_dir'],
        subframes_dir=data_cfg['gopro_subframes_train_dir'],
        patch_size=patch_size,
        epoch_size=data_cfg.get('epoch_size', 50000),
        sharp_ratio=data_cfg.get('sharp_ratio', 0.2),
        frame_parity=data_cfg.get('frame_parity', 'all'),
    )


def create_val_dataset(config):
    data_cfg = config['data']
    return GoProValDataset(
        gopro_dir=data_cfg['gopro_test_dir'],
        subframes_dir=data_cfg['gopro_subframes_test_dir'],
        align_size=32,
        max_per_scene=data_cfg.get('max_val_per_scene', None),
    )


def create_train_loader(train_dataset, batch_size, num_workers):
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)


def create_val_loader(val_dataset, num_workers):
    return DataLoader(val_dataset, batch_size=1, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# --------------------------------------------------------------------------
# Loss weight schedule
# --------------------------------------------------------------------------

def get_loss_weights(global_step, config):
    """Per-group loss weights with optional delay / linear warmup.

    ident   : ramps linearly over training.ident_warmup_steps
    consist : off until training.consist_delay_steps, then ramps over
              training.consist_warmup_steps (0 = switch on at once)
    """
    train_cfg = config['training']
    loss_cfg = config['loss']

    def schedule_factor(delay_steps=0, warmup_steps=0):
        if delay_steps and global_step < delay_steps:
            return 0.0
        rel = global_step - (delay_steps or 0)
        if warmup_steps and warmup_steps > 0 and rel < warmup_steps:
            return rel / warmup_steps
        return 1.0

    f_ident = schedule_factor(
        delay_steps=train_cfg.get('ident_delay_steps', 0),
        warmup_steps=train_cfg.get('ident_warmup_steps', 2000),
    )
    f_consist = schedule_factor(
        delay_steps=train_cfg.get('consist_delay_steps', 0),
        warmup_steps=train_cfg.get('consist_warmup_steps', 0),
    )

    recon_cfg = loss_cfg.get('recon', {})
    ident_cfg = loss_cfg.get('ident', {})
    consist_cfg = loss_cfg.get('consist', {})
    return {
        'recon': {
            'pixel_l1': recon_cfg.get('pixel_l1_weight', 1.0),
            'pixel_lpips': recon_cfg.get('pixel_lpips_weight', 2.0),
            'latent_l2': recon_cfg.get('latent_l2_weight', 0.0),
        },
        'ident': {
            'pixel_l1': f_ident * ident_cfg.get('pixel_l1_weight', 0.0),
            'pixel_lpips': f_ident * ident_cfg.get('pixel_lpips_weight', 0.0),
            'latent_l2': f_ident * ident_cfg.get('latent_l2_weight', 0.0),
        },
        'consist': {
            'pixel_l1': f_consist * consist_cfg.get('pixel_l1_weight', 0.0),
            'latent_l2': f_consist * consist_cfg.get('latent_l2_weight', 0.0),
        },
    }


# --------------------------------------------------------------------------
# Edge-aware LPIPS
# --------------------------------------------------------------------------

def get_edge_map(x):
    """Sobel edge magnitude. x: (B, 3, H, W) -> (B, 1, H, W)."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    gray = x.mean(dim=1, keepdim=True)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)


def ea_lpips(pred, gt, net_lpips):
    """LPIPS(pred, gt) + LPIPS(edge(pred), edge(gt))."""
    loss_lpips = net_lpips(pred, gt).mean()
    edge_pred = get_edge_map(pred)
    edge_gt = get_edge_map(gt)
    loss_edge = net_lpips(edge_pred.repeat(1, 3, 1, 1), edge_gt.repeat(1, 3, 1, 1)).mean()
    return loss_lpips + loss_edge


def perceptual_loss(pred, gt, net_lpips, use_ea_lpips):
    if use_ea_lpips:
        return ea_lpips(pred, gt, net_lpips)
    return net_lpips(pred, gt).mean()


# --------------------------------------------------------------------------
# Loss computation
# --------------------------------------------------------------------------

def compute_loss(model, batch, config, net_lpips, global_step):
    weights = get_loss_weights(global_step, config)
    loss_cfg = config['loss']
    use_ea = loss_cfg.get('use_ea_lpips', False)
    device = next(model.transformer.parameters()).device

    x_blur = batch['x_blur']
    s = batch['s']
    t = batch['t']
    gt_avg = batch['gt_avg']
    gt_boundary_t = batch['gt_boundary_t']
    B = x_blur.shape[0]

    # Sharp samples (s == t) are queried as [t - eps, t + eps]
    sharp_eps = loss_cfg.get('sharp_eps', 0.01)
    is_sharp = (s == t)
    if is_sharp.any():
        s = torch.where(is_sharp, t - sharp_eps, s)
        t = torch.where(is_sharp, t + sharp_eps, t)

    w_recon = weights['recon']
    w_ident = weights['ident']
    w_consist = weights['consist']

    need_recon_latent = w_recon['latent_l2'] > 0
    need_recon_pixel = (w_recon['pixel_l1'] + w_recon['pixel_lpips']) > 0
    need_ident_latent = w_ident['latent_l2'] > 0
    need_ident_pixel = (w_ident['pixel_l1'] + w_ident['pixel_lpips']) > 0

    # ==================== Encode ====================
    z_blur = model.encode(x_blur, mode='mean')

    if need_recon_latent and need_ident_latent:
        z_gt_stack = model.encode(torch.cat([gt_avg, gt_boundary_t], dim=0), mode='mean')
        z_gt, z_gt_boundary_t = z_gt_stack[:B], z_gt_stack[B:]
        del z_gt_stack
    elif need_recon_latent:
        z_gt = model.encode(gt_avg, mode='mean')
        z_gt_boundary_t = None
    elif need_ident_latent:
        z_gt = None
        z_gt_boundary_t = model.encode(gt_boundary_t, mode='mean')
    else:
        z_gt = None
        z_gt_boundary_t = None

    zero = lambda: torch.tensor(0.0, device=device)

    # ==================== Recon: interval mean ====================
    z_pred = model.predict_mean_image(z_blur, s, t)

    recon_latent_l2 = F.mse_loss(z_pred.float(), z_gt.float()) if need_recon_latent else zero()
    recon_pixel_l1, recon_pixel_lpips = zero(), zero()
    if need_recon_pixel:
        pred_pixel = model.decode(z_pred)
        if w_recon['pixel_l1'] > 0:
            recon_pixel_l1 = F.l1_loss(pred_pixel.float(), gt_avg.float())
        if w_recon['pixel_lpips'] > 0:
            recon_pixel_lpips = perceptual_loss(pred_pixel.float(), gt_avg.float(), net_lpips, use_ea)
        del pred_pixel

    total_loss = (w_recon['latent_l2'] * recon_latent_l2
                  + w_recon['pixel_l1'] * recon_pixel_l1
                  + w_recon['pixel_lpips'] * recon_pixel_lpips)

    loss_dict = {
        'recon/latent_l2': recon_latent_l2.detach(),
        'recon/pixel_l1': recon_pixel_l1.detach(),
        'recon/pixel_lpips': recon_pixel_lpips.detach(),
    }

    # ==================== Ident: sharp frame at t ====================
    ident_latent_l2, ident_pixel_l1, ident_pixel_lpips = zero(), zero(), zero()

    if need_ident_latent or need_ident_pixel:
        t_query_s = (t - sharp_eps).clamp(-1, 1)
        t_query_t = (t + sharp_eps).clamp(-1, 1)
        z_sharp_pred = model.predict_mean_image(z_blur, t_query_s, t_query_t)

        if need_ident_latent:
            ident_latent_l2 = F.mse_loss(z_sharp_pred.float(), z_gt_boundary_t.float().detach())

        if need_ident_pixel:
            sharp_pred_pixel = model.decode(z_sharp_pred)
            if w_ident['pixel_l1'] > 0:
                ident_pixel_l1 = F.l1_loss(sharp_pred_pixel.float(), gt_boundary_t.float())
            if w_ident['pixel_lpips'] > 0:
                ident_pixel_lpips = perceptual_loss(
                    sharp_pred_pixel.float(), gt_boundary_t.float(), net_lpips, use_ea)
            del sharp_pred_pixel
        del z_sharp_pred

        total_loss = (total_loss
                      + w_ident['latent_l2'] * ident_latent_l2
                      + w_ident['pixel_l1'] * ident_pixel_l1
                      + w_ident['pixel_lpips'] * ident_pixel_lpips)

    loss_dict['ident/latent_l2'] = ident_latent_l2.detach()
    loss_dict['ident/pixel_l1'] = ident_pixel_l1.detach()
    loss_dict['ident/pixel_lpips'] = ident_pixel_lpips.detach()

    # ==================== Consist: additivity ====================
    # (t-s) f(s,t) = (m-s) f(s,m) + (t-m) f(m,t), m uniform in [s+0.1(t-s), s+0.9(t-s)].
    # The right-hand side is computed without gradients.
    consist_latent_l2, consist_pixel_l1 = zero(), zero()
    need_consist_latent = w_consist['latent_l2'] > 0
    need_consist_pixel = w_consist['pixel_l1'] > 0

    if need_consist_latent or need_consist_pixel:
        rand_ratio = torch.rand_like(s) * 0.8 + 0.1
        m = s + (t - s) * rand_ratio

        z_f_st = z_pred
        with torch.no_grad():
            z_pair = model.predict_mean_image(
                torch.cat([z_blur, z_blur], dim=0),
                torch.cat([s, m], dim=0),
                torch.cat([m, t], dim=0),
            )
            z_f_sm, z_f_mt = z_pair[:B], z_pair[B:]
            del z_pair

        tms = (t - s).view(-1, 1, 1, 1).float()
        mms = (m - s).view(-1, 1, 1, 1).float()
        tmm = (t - m).view(-1, 1, 1, 1).float()

        if need_consist_latent:
            rhs_latent = (mms * z_f_sm.float() + tmm * z_f_mt.float()) / tms
            consist_latent_l2 = F.mse_loss(z_f_st.float(), rhs_latent.detach())
            del rhs_latent

        if need_consist_pixel:
            pix_f_st = model.decode(z_f_st)
            with torch.no_grad():
                pix_f_sm = model.decode(z_f_sm)
                pix_f_mt = model.decode(z_f_mt)
            rhs_pixel = (mms * pix_f_sm.float() + tmm * pix_f_mt.float()) / tms
            consist_pixel_l1 = F.l1_loss(pix_f_st.float(), rhs_pixel.detach())
            del pix_f_st, pix_f_sm, pix_f_mt, rhs_pixel

        del z_f_sm, z_f_mt

        total_loss = (total_loss
                      + w_consist['latent_l2'] * consist_latent_l2
                      + w_consist['pixel_l1'] * consist_pixel_l1)

    loss_dict['consist/latent_l2'] = consist_latent_l2.detach()
    loss_dict['consist/pixel_l1'] = consist_pixel_l1.detach()
    loss_dict['total_loss'] = total_loss.detach()

    return total_loss, loss_dict


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, accelerator, net_lpips):
    """Center-frame deblur metrics + N-frame video PSNR (best of forward/backward)."""
    model.set_eval()
    device = accelerator.device

    sums = {k: 0.0 for k in ('deblur_psnr', 'deblur_ssim', 'deblur_lpips', 'video_psnr', 'pwpsnr')}
    direction_correct = 0
    even_psnr_sum = odd_psnr_sum = 0.0
    even_count_total = odd_count_total = 0
    count = 0

    for batch in val_loader:
        x_blur = batch['x_blur'].to(device)
        gt_sharp = batch['gt_sharp'].to(device)
        gt_frames = batch['gt_frames'].to(device)  # (1, N, 3, H, W)
        N = gt_frames.shape[1]

        # 1. Center frame
        pred_sharp = model.deblur(x_blur)
        sums['deblur_lpips'] += net_lpips(pred_sharp[0:1].float(), gt_sharp[0:1].float()).item()

        pred_np = ((pred_sharp[0] + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
        gt_np = ((gt_sharp[0] + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
        sums['deblur_psnr'] += peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
        sums['deblur_ssim'] += structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=2)

        # 2. N frames (chunked to fit memory at full resolution)
        pred_video = model.extract_video(x_blur, n_frames=N, max_batch=3)
        pred_frames = ((pred_video[0] + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(0, 2, 3, 1)
        gt_frames_np = ((gt_frames[0] + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(0, 2, 3, 1)

        # 3. Bidirectional comparison
        fwd_mse = np.zeros_like(gt_frames_np[0])
        bwd_mse = np.zeros_like(gt_frames_np[0])
        fwd_psnr_per = np.zeros(N)
        bwd_psnr_per = np.zeros(N)
        for fi in range(N):
            fwd_mse += (pred_frames[fi] - gt_frames_np[fi]) ** 2
            bwd_mse += (pred_frames[fi] - gt_frames_np[N - 1 - fi]) ** 2
            fwd_psnr_per[fi] = peak_signal_noise_ratio(gt_frames_np[fi], pred_frames[fi], data_range=1.0)
            bwd_psnr_per[fi] = peak_signal_noise_ratio(gt_frames_np[N - 1 - fi], pred_frames[fi], data_range=1.0)

        fwd_psnr_avg = fwd_psnr_per.mean()
        bwd_psnr_avg = bwd_psnr_per.mean()
        sums['video_psnr'] += max(fwd_psnr_avg, bwd_psnr_avg)

        # PSNR split by parity of the matched GT sub-frame index
        if fwd_psnr_avg >= bwd_psnr_avg:
            direction_correct += 1
            psnr_per, gt_idx = fwd_psnr_per, np.arange(N)
        else:
            psnr_per, gt_idx = bwd_psnr_per, np.arange(N)[::-1]
        even_mask = (gt_idx % 2 == 0)
        odd_mask = ~even_mask
        if even_mask.any():
            even_psnr_sum += psnr_per[even_mask].sum()
            even_count_total += int(even_mask.sum())
        if odd_mask.any():
            odd_psnr_sum += psnr_per[odd_mask].sum()
            odd_count_total += int(odd_mask.sum())

        pw_mse = np.minimum(fwd_mse / N, bwd_mse / N)
        sums['pwpsnr'] += 10 * np.log10(1.0 / np.mean(pw_mse).clip(min=1e-10))

        count += 1
        del x_blur, gt_sharp, gt_frames, pred_sharp, pred_video

    n = max(count, 1)
    model.set_train()
    out = {f'val/{k}': v / n for k, v in sums.items()}
    out.update({
        'val/direction_acc': direction_correct / n * 100,
        'val/even_psnr': even_psnr_sum / max(even_count_total, 1),
        'val/odd_psnr': odd_psnr_sum / max(odd_count_total, 1),
        'val/num_images': count,
    })
    return out


@torch.no_grad()
def save_val_images(model, val_loader, accelerator, global_step, output_dir, num_images=8):
    """Save a few blur / prediction / GT triplets and 5-frame videos."""
    model.set_eval()
    device = accelerator.device

    samples_dir = Path(output_dir) / "samples" / f"step_{global_step:06d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    def to_pil(x):
        return TF.to_pil_image(((x + 1.0) / 2.0).clamp(0, 1).float().cpu())

    count = 0
    for batch in val_loader:
        if count >= num_images:
            break
        x_blur = batch['x_blur'].to(device)
        gt_sharp = batch['gt_sharp'].to(device)
        base_name = Path(batch.get('filename', [f'img_{count}'])[0]).stem

        pred_sharp = model.deblur(x_blur)
        to_pil(x_blur[0]).save(samples_dir / f"{base_name}_blur.png")
        to_pil(pred_sharp[0]).save(samples_dir / f"{base_name}_pred.png")
        to_pil(gt_sharp[0]).save(samples_dir / f"{base_name}_gt.png")

        if count < 3:
            frames = model.extract_video(x_blur, n_frames=5)
            for fi in range(5):
                to_pil(frames[0, fi]).save(samples_dir / f"{base_name}_frame{fi}.png")
            del frames

        del x_blur, gt_sharp, pred_sharp
        count += 1

    logger.info(f"Saved {count} images to {samples_dir}")
    model.set_train()
    torch.cuda.empty_cache()


def save_unwrapped(model, accelerator, save_path):
    """Unwrap the accelerate-prepared modules, save, then restore."""
    orig_transformer = model.transformer
    orig_cond = model.null_conditioner
    model.transformer = accelerator.unwrap_model(model.transformer)
    model.null_conditioner = accelerator.unwrap_model(model.null_conditioner)
    model.save_model(save_path)
    model.transformer = orig_transformer
    model.null_conditioner = orig_cond


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(config, args):
    train_cfg = config['training']
    log_cfg = config['logging']
    data_cfg = config['data']

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
        mixed_precision=train_cfg.get('mixed_precision', 'bf16'),
        log_with="wandb" if args.wandb else None,
    )
    if args.seed is not None:
        set_seed(args.seed)

    output_dir = Path(args.output_dir or log_cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    if args.wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project=log_cfg.get('wandb_project', 'pickmoment'),
                   name=output_dir.name, config=config, dir=str(output_dir))

    # ---- Model ----
    model_cfg = config['model']
    model_args = PickMomentArgs(
        pretrained_model_name_or_path=model_cfg['backbone'],
        num_cond_tokens=model_cfg.get('num_cond_tokens', 4),
        enable_grad_checkpointing=model_cfg.get('enable_grad_checkpointing', True),
    )
    model = PickMoment(model_args, mode='train')
    model.set_train()

    # ---- Data ----
    current_patch_size, current_batch_size = get_progressive_config(0, config)
    current_stage = get_stage_for_step(0, config)
    progressive_enabled = data_cfg.get('progressive', {}).get('enabled', False)
    if progressive_enabled:
        logger.info("Progressive patch schedule:")
        for i, (ps, bs, until) in enumerate(data_cfg['progressive']['schedule']):
            logger.info(f"  Stage {i}: patch={ps}, batch={bs}, until step {until}")

    num_workers = data_cfg.get('num_workers', 4)
    train_dataset = create_train_dataset(config, current_patch_size)
    train_loader = create_train_loader(train_dataset, current_batch_size, num_workers)
    val_loader = create_val_loader(create_val_dataset(config), num_workers)

    # ---- Optimizer / scheduler ----
    params = model.get_trainable_params()
    opt_kwargs = train_cfg.get('optimizer_kwargs', {})
    logger.info(f"Trainable parameters: {sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(
        params,
        lr=train_cfg['learning_rate'],
        betas=opt_kwargs.get('betas', [0.9, 0.999]),
        weight_decay=opt_kwargs.get('weight_decay', 0.01),
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_steps = train_cfg.get('warmup_steps', 1000)
    max_steps = train_cfg['max_steps']
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps,
                              eta_min=train_cfg['learning_rate'] * train_cfg.get('eta_min_ratio', 0.01)),
        ],
        milestones=[warmup_steps],
    )

    model.transformer, model.null_conditioner, optimizer, train_loader, val_loader, scheduler = (
        accelerator.prepare(model.transformer, model.null_conditioner,
                            optimizer, train_loader, val_loader, scheduler)
    )
    model.vae = model.vae.to(accelerator.device)

    # ---- LPIPS (loss) ----
    net_lpips = lpips.LPIPS(net=config['loss'].get('lpips_net', 'alex')).to(accelerator.device)
    net_lpips.requires_grad_(False)
    net_lpips = accelerator.prepare(net_lpips)

    # ---- Resume (weights only) ----
    global_step = 0
    if args.resume:
        logger.info(f"Loading weights from {args.resume}")
        orig_transformer, orig_cond = model.transformer, model.null_conditioner
        model.transformer = accelerator.unwrap_model(model.transformer)
        model.null_conditioner = accelerator.unwrap_model(model.null_conditioner)
        model.load_ckpt(args.resume)
        model.transformer, model.null_conditioner = orig_transformer, orig_cond

    # ---- Loop ----
    logger.info(f"Starting PickMoment training for {max_steps} steps")
    best_psnr = 0.0
    train_iter = iter(train_loader)
    progress_bar = tqdm(total=max_steps, initial=global_step,
                        disable=not accelerator.is_local_main_process, desc="Training")

    while global_step < max_steps:
        with accelerator.accumulate(model.transformer):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            loss, loss_dict = compute_loss(model, batch, config, net_lpips, global_step)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                all_params = list(model.transformer.parameters()) + list(model.null_conditioner.parameters())
                trainable = [p for p in all_params if p.requires_grad]
                accelerator.clip_grad_norm_(trainable, train_cfg.get('grad_clip', 1.0))

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        del batch, loss

        if not accelerator.sync_gradients:
            continue
        global_step += 1
        progress_bar.update(1)

        # ---- Logging ----
        if global_step % log_cfg.get('log_interval', 50) == 0 and accelerator.is_main_process:
            log_dict = {'train/lr': optimizer.param_groups[0]['lr'], 'train/step': global_step}
            for k, v in loss_dict.items():
                log_dict[f'train/{k}'] = v.item() if isinstance(v, torch.Tensor) else v
            if wandb is not None:
                wandb.log(log_dict, step=global_step)

            postfix = {'lat': f"{loss_dict['recon/latent_l2'].item():.4f}"}
            ident_total = sum(loss_dict[k].item() for k in
                              ('ident/latent_l2', 'ident/pixel_l1', 'ident/pixel_lpips'))
            consist_total = sum(loss_dict[k].item() for k in ('consist/latent_l2', 'consist/pixel_l1'))
            if ident_total > 0:
                postfix['ident'] = f"{ident_total:.4f}"
            if consist_total > 0:
                postfix['cons'] = f"{consist_total:.4f}"
            if progressive_enabled:
                postfix['patch'] = current_patch_size
            progress_bar.set_postfix(**postfix)

        # ---- Progressive patch stage transition ----
        if progressive_enabled:
            new_stage = get_stage_for_step(global_step, config)
            if new_stage != current_stage:
                current_stage = new_stage
                current_patch_size, current_batch_size = get_progressive_config(global_step, config)
                logger.info(f"[Step {global_step}] Stage {current_stage}: "
                            f"patch={current_patch_size}, batch={current_batch_size}")

                del train_dataset, train_loader, train_iter
                torch.cuda.empty_cache()
                train_dataset = create_train_dataset(config, current_patch_size)
                train_loader = accelerator.prepare(
                    create_train_loader(train_dataset, current_batch_size, num_workers))
                train_iter = iter(train_loader)

                if wandb is not None:
                    wandb.log({'train/patch_size': current_patch_size,
                               'train/batch_size': current_batch_size,
                               'train/stage': current_stage}, step=global_step)

        # ---- Validation ----
        if global_step % train_cfg.get('validation_steps', 1000) == 0:
            torch.cuda.empty_cache()
            val_metrics = validate(model, val_loader, accelerator, net_lpips)

            if accelerator.is_main_process:
                logger.info(
                    f"Step {global_step}: "
                    f"Deblur={val_metrics['val/deblur_psnr']:.2f}dB "
                    f"SSIM={val_metrics['val/deblur_ssim']:.4f} "
                    f"LPIPS={val_metrics['val/deblur_lpips']:.4f} "
                    f"Video={val_metrics['val/video_psnr']:.2f}dB "
                    f"PWPSNR={val_metrics['val/pwpsnr']:.2f}dB "
                    f"Even={val_metrics['val/even_psnr']:.2f}dB "
                    f"Odd={val_metrics['val/odd_psnr']:.2f}dB"
                )
                if wandb is not None:
                    wandb.log(val_metrics, step=global_step)

                if val_metrics['val/deblur_psnr'] > best_psnr:
                    best_psnr = val_metrics['val/deblur_psnr']
                    save_unwrapped(model, accelerator, output_dir / "best_model.pt")
                    logger.info(f"New best deblur PSNR: {best_psnr:.2f} dB")

                if log_cfg.get('save_images', True):
                    save_val_images(model, val_loader, accelerator, global_step, output_dir,
                                    num_images=log_cfg.get('num_val_images', 8))

            model.set_train()

        # ---- Periodic checkpoint ----
        if global_step % train_cfg.get('save_steps', 5000) == 0 and accelerator.is_main_process:
            save_unwrapped(model, accelerator, output_dir / f"checkpoint_{global_step:06d}.pt")

    progress_bar.close()
    if accelerator.is_main_process:
        save_unwrapped(model, accelerator, output_dir / "final_model.pt")
        logger.info(f"Training complete. Best deblur PSNR: {best_psnr:.2f} dB")
    if wandb is not None:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="PickMoment training")
    parser.add_argument('--config', type=str, default='configs/train.yaml')
    parser.add_argument('--output_dir', type=str, default=None, help="Override logging.output_dir")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                        help="Initialize weights from a checkpoint (optimizer state is not restored)")
    parser.add_argument('--wandb', action='store_true', help="Log to Weights & Biases")
    args = parser.parse_args()
    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
