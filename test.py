"""
PickMoment evaluation: single-image deblurring (Tab. 1) and GoPro per-frame video.

Modes:
    gopro_deblur      GoPro test set, center frame. PSNR / SSIM / LPIPS / DISTS
    hide_deblur       HIDE test set, center frame. PSNR / SSIM / LPIPS / DISTS
    realblur_j_deblur RealBlur-J, ECC-aligned (official RealBlur protocol)
    realblur_r_deblur RealBlur-R, ECC-aligned (official RealBlur protocol)
    gopro_perframe    GoPro-N (default N=11) per-frame PSNR / SSIM of the
                      extracted N-frame video, best of forward/backward order

Blur-to-video on GoPro-7 (Table 2) is in test_blur2vid.py.

Usage:
    python test.py --config configs/test.yaml --mode gopro_deblur --output_dir results/gopro
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from eval_metrics import (calc_psnr_rgb, calc_ssim_3d, to_uint8,
                          ecc_align, calc_psnr_masked, calc_ssim_masked)
from models import build_model
from dataloader import GoProTestDataset, HIDETestDataset, RealBlurTestDataset


# ============================================================
# Utils
# ============================================================

def pad_to_align(tensor, align_size=32):
    """Reflection-pad (B, C, H, W) so H and W are multiples of align_size."""
    _, _, h, w = tensor.shape
    pad_h = (align_size - h % align_size) % align_size
    pad_w = (align_size - w % align_size) % align_size
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, (pad_h, pad_w)


def unpad(tensor, pad_hw):
    """Remove padding from (B, C, H, W) or (B, N, C, H, W)."""
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return tensor
    if tensor.dim() == 5:
        return tensor[:, :, :, :tensor.shape[3] - pad_h, :tensor.shape[4] - pad_w]
    return tensor[:, :, :tensor.shape[2] - pad_h, :tensor.shape[3] - pad_w]


def build_perceptual_nets(device):
    import lpips as lpips_lib
    from DISTS_pytorch import DISTS
    net_lpips = lpips_lib.LPIPS(net='alex').to(device)
    net_lpips.requires_grad_(False)
    net_dists = DISTS().to(device)
    net_dists.requires_grad_(False)
    return net_lpips, net_dists


def write_metrics(path, label, n, psnr, ssim, lpips_v, dists_v):
    lines = [
        f"{label}",
        f"Images: {n}",
        f"PSNR:   {psnr:.2f} dB",
        f"SSIM:   {ssim:.4f}",
        f"LPIPS:  {lpips_v:.4f}",
        f"DISTS:  {dists_v:.4f}",
    ]
    print("=" * 60)
    print(lines[0])
    for line in lines[1:]:
        print("  " + line)
    print("=" * 60)
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Metrics saved to {path}")


def make_loader(dataset):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


# ============================================================
# Deblurring: GoPro / HIDE (RGB uint8 PSNR, 3D-Gaussian SSIM)
# ============================================================

@torch.no_grad()
def _test_deblur_dataset(model, test_dataset, output_dir, label, metrics_filename,
                         eps, has_gt=True):
    """Center-frame deblurring on (x_blur, gt_sharp) pairs."""
    test_loader = make_loader(test_dataset)
    device = next(model.transformer.parameters()).device
    if has_gt:
        net_lpips, net_dists = build_perceptual_nets(device)

    deblur_dir = Path(output_dir) / "deblur"
    deblur_dir.mkdir(parents=True, exist_ok=True)

    psnr_list, ssim_list, lpips_list, dists_list = [], [], [], []

    for batch in tqdm(test_loader, desc=label):
        x_blur = batch['x_blur'].to(device)
        gt_sharp = batch['gt_sharp'].to(device)
        filename = batch['filename'][0]

        x_blur_pad, pad_hw = pad_to_align(x_blur, align_size=32)
        pred_sharp = model.deblur(x_blur_pad, eps=eps)
        pred_sharp = unpad(pred_sharp, pad_hw)

        pred_01 = (pred_sharp + 1.0) / 2.0
        gt_01 = (gt_sharp + 1.0) / 2.0
        pred_np = pred_01[0].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)

        if has_gt:
            lpips_list.append(net_lpips(pred_sharp.float(), gt_sharp.float()).item())
            dists_list.append(net_dists(pred_01.float(), gt_01.float()).item())

            gt_np = gt_01[0].clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
            pred_u8 = to_uint8(pred_np)
            gt_u8 = to_uint8(gt_np)
            psnr_list.append(calc_psnr_rgb(pred_u8, gt_u8))
            ssim_list.append(calc_ssim_3d(pred_u8, gt_u8))

        Image.fromarray((pred_np * 255).clip(0, 255).astype(np.uint8)).save(
            deblur_dir / f"{filename}.png"
        )

        del x_blur, gt_sharp, pred_sharp

    if has_gt:
        write_metrics(Path(output_dir) / metrics_filename, f"{label} Results",
                      len(psnr_list), np.mean(psnr_list), np.mean(ssim_list),
                      np.mean(lpips_list), np.mean(dists_list))
    else:
        print(f"{label}: no GT found, saved predictions only.")


def test_gopro_deblur(model, config, output_dir, eps):
    data_cfg = config['data']
    test_cfg = config.get('test', {})
    ds = GoProTestDataset(
        gopro_dir=data_cfg['gopro_test_dir'],
        subframes_dir=data_cfg['gopro_subframes_test_dir'],
        max_per_scene=test_cfg.get('max_per_scene'),
    )
    _test_deblur_dataset(model, ds, output_dir, label="GoPro Deblur",
                         metrics_filename="metrics_gopro.txt", eps=eps)


def test_hide_deblur(model, config, output_dir, eps):
    ds = HIDETestDataset(hide_dir=config['data']['hide_test_dir'])
    _test_deblur_dataset(model, ds, output_dir, label="HIDE Deblur",
                         metrics_filename="metrics_hide.txt", eps=eps, has_gt=ds.has_gt)


# ============================================================
# Deblurring: RealBlur (ECC alignment + masked PSNR/SSIM)
# ============================================================

@torch.no_grad()
def test_realblur_deblur(model, config, output_dir, subset, eps):
    """RealBlur-J / RealBlur-R with the official ECC alignment protocol.

    Based on https://github.com/rimchang/RealBlur/blob/master/evaluation_RealBlur_ecc.py
    """
    base_dir = config['data']['realblur_test_dir']
    test_dataset = RealBlurTestDataset(realblur_dir=f"{base_dir}/RealBlur-{subset}")
    label = f"RealBlur-{subset} Deblur"

    test_loader = make_loader(test_dataset)
    device = next(model.transformer.parameters()).device
    net_lpips, net_dists = build_perceptual_nets(device)

    deblur_dir = Path(output_dir) / "deblur"
    deblur_dir.mkdir(parents=True, exist_ok=True)

    psnr_list, ssim_list, lpips_list, dists_list = [], [], [], []

    for batch in tqdm(test_loader, desc=label):
        x_blur = batch['x_blur'].to(device)
        gt_sharp = batch['gt_sharp'].to(device)
        filename = batch['filename'][0]

        x_blur_pad, pad_hw = pad_to_align(x_blur, align_size=32)
        pred_sharp = model.deblur(x_blur_pad, eps=eps)
        pred_sharp = unpad(pred_sharp, pad_hw)

        pred_np_raw = ((pred_sharp[0] + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
        Image.fromarray((pred_np_raw * 255).clip(0, 255).astype(np.uint8)).save(
            deblur_dir / f"{filename}.png"
        )

        # ECC alignment + intensity matching on the float prediction
        pred_np = pred_np_raw.astype(np.float32)
        gt_np = ((gt_sharp[0] + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
        aligned_pred, aligned_gt, mask = ecc_align(pred_np, gt_np)

        psnr_list.append(calc_psnr_masked(aligned_gt, aligned_pred, mask, data_range=1.0))
        ssim_list.append(calc_ssim_masked(aligned_gt, aligned_pred, mask, data_range=1.0))

        aligned_pred_t = torch.from_numpy(aligned_pred).permute(2, 0, 1).unsqueeze(0).to(device)
        aligned_gt_t = torch.from_numpy(aligned_gt).permute(2, 0, 1).unsqueeze(0).to(device)
        lpips_list.append(net_lpips((aligned_pred_t * 2.0 - 1.0).float(),
                                    (aligned_gt_t * 2.0 - 1.0).float()).item())
        dists_list.append(net_dists(aligned_pred_t.float(), aligned_gt_t.float()).item())

        del x_blur, gt_sharp, pred_sharp

    write_metrics(Path(output_dir) / f"metrics_realblur_{subset.lower()}.txt",
                  f"{label} Results (ECC-aligned)", len(test_dataset),
                  float(np.mean(psnr_list)), float(np.mean(ssim_list)),
                  float(np.mean(lpips_list)), float(np.mean(dists_list)))


# ============================================================
# GoPro-N per-frame video evaluation (full resolution)
# ============================================================

@torch.no_grad()
def test_gopro_perframe(model, config, output_dir, eps, n_frames):
    """Per-frame PSNR / SSIM of the extracted N-frame video.

    Uses only GoPro test scenes with N averaged frames. For each clip the
    direction (forward or reversed GT order) with the higher mean PSNR is used.
    """
    data_cfg = config['data']
    test_cfg = config.get('test', {})

    test_dataset = GoProTestDataset(
        gopro_dir=data_cfg['gopro_test_dir'],
        subframes_dir=data_cfg['gopro_subframes_test_dir'],
        filter_n=n_frames,
        max_per_scene=test_cfg.get('max_per_scene'),
    )
    if len(test_dataset) == 0:
        raise RuntimeError(f"No clips with N={n_frames} in {data_cfg['gopro_test_dir']}")
    test_loader = make_loader(test_dataset)
    device = next(model.transformer.parameters()).device

    psnr_sum = np.zeros(n_frames, dtype=np.float64)
    ssim_sum = np.zeros(n_frames, dtype=np.float64)
    n_clips = 0
    n_fwd = 0

    video_dir = Path(output_dir) / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(test_loader, desc=f"GoPro-{n_frames} per-frame"):
        x_blur = batch['x_blur'].to(device)
        gt_frames = batch['gt_frames'].to(device)  # (1, N, 3, H, W)
        N = gt_frames.shape[1]
        filename = batch['filename'][0]
        assert N == n_frames, f"clip {filename} has N={N}, expected {n_frames}"

        x_blur_pad, pad_hw = pad_to_align(x_blur, align_size=32)
        pred_video = model.extract_video(x_blur_pad, n_frames=N, eps=eps)
        pred_video = unpad(pred_video, pad_hw)

        pred_np = ((pred_video[0] + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(0, 2, 3, 1)
        gt_np = ((gt_frames[0] + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(0, 2, 3, 1)

        fwd_psnr = np.array([peak_signal_noise_ratio(gt_np[i], pred_np[i], data_range=1.0)
                             for i in range(N)])
        bwd_psnr = np.array([peak_signal_noise_ratio(gt_np[N - 1 - i], pred_np[i], data_range=1.0)
                             for i in range(N)])
        fwd_ssim = np.array([structural_similarity(gt_np[i], pred_np[i], data_range=1.0, channel_axis=2)
                             for i in range(N)])
        bwd_ssim = np.array([structural_similarity(gt_np[N - 1 - i], pred_np[i], data_range=1.0, channel_axis=2)
                             for i in range(N)])

        if fwd_psnr.mean() >= bwd_psnr.mean():
            psnr_sum += fwd_psnr
            ssim_sum += fwd_ssim
            n_fwd += 1
        else:
            psnr_sum += bwd_psnr
            ssim_sum += bwd_ssim

        sample_video_dir = video_dir / filename
        sample_video_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(N):
            Image.fromarray((pred_np[fi] * 255).clip(0, 255).astype(np.uint8)).save(
                sample_video_dir / f"frame_{fi:05d}.png"
            )

        n_clips += 1
        del x_blur, gt_frames, pred_video

    psnr_per_frame = (psnr_sum / max(n_clips, 1)).tolist()
    ssim_per_frame = (ssim_sum / max(n_clips, 1)).tolist()
    avg_psnr = float(np.mean(psnr_per_frame))
    avg_ssim = float(np.mean(ssim_per_frame))

    print("=" * 60)
    print(f"GoPro-{n_frames} per-frame video results (full resolution)")
    print("=" * 60)
    print(f"  Clips: {n_clips}  (forward: {n_fwd}, backward: {n_clips - n_fwd})")
    print(f"  {'Frame':<8}{'PSNR':>10}{'SSIM':>10}")
    for i, (p, s) in enumerate(zip(psnr_per_frame, ssim_per_frame)):
        print(f"  {i:<8}{p:10.2f}{s:10.4f}")
    print(f"  {'Avg':<8}{avg_psnr:10.2f}{avg_ssim:10.4f}")
    print("=" * 60)

    metrics_path = Path(output_dir) / f"metrics_gopro{n_frames}_perframe.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"GoPro-{n_frames} per-frame video results (full resolution)\n")
        f.write(f"Clips: {n_clips}  (forward: {n_fwd}, backward: {n_clips - n_fwd})\n")
        f.write(f"Frames per clip: {n_frames}\n\n")
        f.write(f"{'frame':<8}{'PSNR':>10}{'SSIM':>10}\n")
        for i, (p, s) in enumerate(zip(psnr_per_frame, ssim_per_frame)):
            f.write(f"{i:<8}{p:10.4f}{s:10.4f}\n")
        f.write(f"\n{'Avg':<8}{avg_psnr:10.4f}{avg_ssim:10.4f}\n")

    csv_path = Path(output_dir) / f"metrics_gopro{n_frames}_perframe.csv"
    with open(csv_path, 'w') as f:
        f.write("frame_idx,psnr,ssim\n")
        for i, (p, s) in enumerate(zip(psnr_per_frame, ssim_per_frame)):
            f.write(f"{i},{p:.6f},{s:.6f}\n")
    print(f"Metrics saved to {metrics_path} and {csv_path}")


# ============================================================
# Main
# ============================================================

MODES = ['gopro_deblur', 'hide_deblur', 'realblur_j_deblur', 'realblur_r_deblur',
         'gopro_perframe']
MODE_ALIASES = {'gopro_orig_deblur': 'gopro_deblur', 'gopro11_perframe': 'gopro_perframe'}


def parse_args():
    parser = argparse.ArgumentParser(description="PickMoment evaluation")
    parser.add_argument("--config", type=str, default="configs/test.yaml")
    parser.add_argument("--mode", type=str, default=None, choices=MODES + list(MODE_ALIASES),
                        help="Override test.mode in config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override test.output_dir in config")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_model.pt. Overrides config")
    parser.add_argument("--max_per_scene", type=int, default=None,
                        help="Limit GoPro samples per scene (quick sanity check)")
    parser.add_argument("--eps", type=float, default=None,
                        help="Half-width of the (s, t) query around each moment. "
                             "Default: test.eps in config (0.01)")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="N for gopro_perframe (default: test.perframe_n_frames or 11)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_cfg = config['model']
    test_cfg = config.setdefault('test', {})

    if args.mode is not None:
        test_cfg['mode'] = args.mode
    if args.output_dir is not None:
        test_cfg['output_dir'] = args.output_dir
    if args.checkpoint is not None:
        model_cfg['checkpoint'] = args.checkpoint
    if args.max_per_scene is not None:
        test_cfg['max_per_scene'] = args.max_per_scene

    eps = args.eps if args.eps is not None else float(test_cfg.get('eps', 0.01))
    n_frames = args.n_frames or int(test_cfg.get('perframe_n_frames', 11))
    mode = test_cfg.get('mode', 'gopro_deblur')
    mode = MODE_ALIASES.get(mode, mode)
    output_dir = test_cfg.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[test] mode={mode}  eps={eps}  output_dir={output_dir}")

    checkpoint = model_cfg.get('checkpoint')
    if checkpoint is None:
        raise ValueError("model.checkpoint not set (config or --checkpoint)")
    model = build_model(model_cfg, checkpoint=checkpoint, mode='test')
    model.set_eval()

    if mode == 'gopro_deblur':
        test_gopro_deblur(model, config, output_dir, eps)
    elif mode == 'hide_deblur':
        test_hide_deblur(model, config, output_dir, eps)
    elif mode == 'realblur_j_deblur':
        test_realblur_deblur(model, config, output_dir, 'J', eps)
    elif mode == 'realblur_r_deblur':
        test_realblur_deblur(model, config, output_dir, 'R', eps)
    elif mode == 'gopro_perframe':
        test_gopro_perframe(model, config, output_dir, eps, n_frames)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from {MODES}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
