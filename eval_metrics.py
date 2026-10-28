"""
Standalone metric evaluation from saved deblurred images.

Fixed protocol (AdaRevD / FideDiff convention, used by most recent
deblurring papers):

  PSNR : RGB uint8, data_range=255 (skimage.psnr)
  SSIM : AdaRevD 3D-Gaussian kernel (_ssim_3d, auto max_value)
          Ref: https://github.com/INVOKERer/AdaRevD/blob/master/basicsr/metrics/psnr_ssim.py
  LPIPS: alex backbone, full resolution, [-1, 1] input
  DISTS: full resolution, [0, 1] input

Datasets:
  gopro_orig : GOPRO_Large_subframes test (center frame GT)
  hide       : HIDE_test
  realblur_j : RealBlur-J (ECC alignment, masked PSNR/SSIM per RealBlur official)
  realblur_r : RealBlur-R (ECC alignment, masked PSNR/SSIM per RealBlur official)

These are the same metric functions used by test.py. This script lets you
re-score a folder of saved predictions (e.g. {output_dir}/deblur from test.py).
Note: test.py scores the in-memory prediction (rounded to uint8 for GoPro and
HIDE, float for RealBlur), while saved PNGs are truncated to uint8. Re-scoring
saved PNGs here can therefore differ slightly from the test.py numbers.

Usage:
  python eval_metrics.py --dataset gopro_orig --pred_dir results/gopro/deblur
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import cv2
import torch
import torch.nn as nn


# ============================================================
# SSIM: AdaRevD custom 3D-Gaussian kernel
# ============================================================

_SSIM_CONV3D = None


def _get_ssim_conv3d(device):
    """Lazily build an 11x11x11 Gaussian Conv3d on the target device."""
    global _SSIM_CONV3D
    if _SSIM_CONV3D is None or _SSIM_CONV3D.weight.device != device:
        k = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(k, k.T)
        k3 = cv2.getGaussianKernel(11, 1.5)
        kernel = torch.tensor(np.stack([window * v for v in k3], axis=0))
        conv = nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5),
                         bias=False, padding_mode='replicate')
        conv.weight.requires_grad = False
        conv.weight[0, 0, :, :, :] = kernel
        _SSIM_CONV3D = conv.to(device)
    return _SSIM_CONV3D


def _gauss3d(img, conv3d):
    return conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)


def calc_ssim_3d(img1, img2, device=None):
    """AdaRevD-style SSIM with a 3D Gaussian kernel over (H, W, C).

    Accepts uint8 or float arrays with the same shape; max_value is inferred
    from img1.max() (1.0 for [0,1] float, 255.0 otherwise).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conv3d = _get_ssim_conv3d(device)
    img1 = np.asarray(img1).astype(np.float64)
    img2 = np.asarray(img2).astype(np.float64)
    max_value = 1.0 if img1.max() <= 1 else 255.0
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    t1 = torch.tensor(img1).float().to(device)
    t2 = torch.tensor(img2).float().to(device)
    with torch.no_grad():
        mu1 = _gauss3d(t1, conv3d)
        mu2 = _gauss3d(t2, conv3d)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        sig1 = _gauss3d(t1 ** 2, conv3d) - mu1_sq
        sig2 = _gauss3d(t2 ** 2, conv3d) - mu2_sq
        sig12 = _gauss3d(t1 * t2, conv3d) - mu12
        smap = ((2 * mu12 + c1) * (2 * sig12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sig1 + sig2 + c2)
        )
    return float(smap.mean())


def calc_psnr_rgb(img1, img2):
    """RGB uint8 PSNR with data_range=255 (skimage)."""
    return sk_psnr(img2, img1, data_range=255)


def to_uint8(arr_01):
    """Convert a float [0, 1] HWC image array to uint8 (for PSNR/SSIM input)."""
    return (np.clip(arr_01, 0, 1) * 255).round().astype(np.uint8)


# ============================================================
# RealBlur ECC alignment + masked PSNR/SSIM (RealBlur official protocol)
# ============================================================

def ecc_align(deblurred, gt):
    """RealBlur official ECC alignment + intensity matching."""
    z = deblurred.astype(np.float32)
    x = gt.astype(np.float32)

    denom = float(np.sum(z * z))
    scale = (float(np.sum(x * z)) / denom) if denom > 1e-12 else 1.0
    zs = scale * z

    warp_mode = cv2.MOTION_HOMOGRAPHY
    warp_matrix = np.eye(3, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0)

    try:
        gray_x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        gray_zs = cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY)
        (cc, warp_matrix) = cv2.findTransformECC(
            gray_x, gray_zs, warp_matrix, warp_mode, criteria,
            inputMask=None, gaussFiltSize=5,
        )
    except cv2.error:
        return zs, x, np.ones_like(x, dtype='float32')

    target_shape = x.shape
    zr = cv2.warpPerspective(
        zs, warp_matrix, (target_shape[1], target_shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    cr = cv2.warpPerspective(
        np.ones_like(zs, dtype='float32'),
        warp_matrix, (target_shape[1], target_shape[0]),
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    zr = zr * cr
    xr = x * cr
    return zr, xr, cr


def calc_psnr_masked(img_true, img_test, mask, data_range=1.0):
    err = np.sum((img_true - img_test) ** 2, dtype=np.float64) / np.sum(mask)
    return 10 * np.log10((data_range ** 2) / err)


def calc_ssim_masked(img_true, img_test, mask, data_range=1.0):
    _, ssim_map = sk_ssim(
        img_true, img_test,
        multichannel=True, gaussian_weights=True,
        use_sample_covariance=False, data_range=data_range,
        full=True, channel_axis=2,
    )
    ssim_map = ssim_map * mask
    r = int(3.5 * 1.5 + 0.5)
    pad = (2 * r + 1 - 1) // 2
    ssim_crop = ssim_map[pad:-pad, pad:-pad, :]
    mask_crop = mask[pad:-pad, pad:-pad, :]
    ssim = ssim_crop.sum(axis=0).sum(axis=0) / mask_crop.sum(axis=0).sum(axis=0).clip(min=1e-10)
    return float(np.mean(ssim))


# ============================================================
# Dataset GT loaders
# ============================================================

def load_gopro_orig_pairs(pred_dir, gopro_dir, subframes_dir):
    """Load pred/GT pairs for the GoPro test set (center sub-frame as GT)."""
    pred_dir = Path(pred_dir)
    gopro_dir = Path(gopro_dir)
    subframes_dir = Path(subframes_dir)

    pairs = []
    for scene_dir in sorted(gopro_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name
        sf_scene = subframes_dir / scene_name
        if not sf_scene.exists():
            continue

        n_orig = None
        for txt in scene_dir.glob("frames *.txt"):
            parts = txt.stem.split()
            if len(parts) >= 2:
                n_orig = int(parts[1])
                break
        if n_orig is None:
            for part in scene_name.split("_"):
                if part.isdigit() and int(part) in (7, 9, 11, 13, 15):
                    n_orig = int(part)
                    break
        if n_orig is None:
            continue

        sf_paths = sorted(sf_scene.glob("*.png"))
        n_blur = len(sf_paths) // n_orig

        for blur_idx in range(n_blur):
            center = blur_idx * n_orig + n_orig // 2
            name = f"{scene_name}_seq{blur_idx:04d}"
            pred_path = pred_dir / f"{name}.png"
            if pred_path.exists():
                pairs.append({"pred": str(pred_path), "gt": str(sf_paths[center]), "name": name})

    return pairs


def load_paired_dir(pred_dir, gt_dir):
    """Load pred/GT pairs from directories with matching filenames."""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    pairs = []
    for pred_path in sorted(pred_dir.glob("*.png")):
        gt_path = gt_dir / pred_path.name
        if gt_path.exists():
            pairs.append({"pred": str(pred_path), "gt": str(gt_path), "name": pred_path.stem})
    return pairs


# ============================================================
# Main evaluation
# ============================================================

def evaluate(pairs, use_ecc=False, use_lpips=True, use_dists=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_lpips, net_dists = None, None
    if use_lpips:
        import lpips as lpips_lib
        net_lpips = lpips_lib.LPIPS(net='alex').to(device)
        net_lpips.requires_grad_(False)
    if use_dists:
        from DISTS_pytorch import DISTS
        net_dists = DISTS().to(device)
        net_dists.requires_grad_(False)

    psnr_list, ssim_list, lpips_list, dists_list = [], [], [], []

    for p in tqdm(pairs, desc="Evaluating"):
        pred = np.array(Image.open(p["pred"]).convert("RGB"))
        gt = np.array(Image.open(p["gt"]).convert("RGB"))
        assert pred.shape == gt.shape, \
            f"Size mismatch: pred {pred.shape} vs gt {gt.shape} for {p['name']}"

        if use_ecc:
            pred_f = pred.astype(np.float32) / 255.0
            gt_f = gt.astype(np.float32) / 255.0
            aligned_pred, aligned_gt, mask = ecc_align(pred_f, gt_f)

            # RealBlur official: masked PSNR/SSIM on ECC-aligned pair
            psnr_list.append(calc_psnr_masked(aligned_gt, aligned_pred, mask, data_range=1.0))
            ssim_list.append(calc_ssim_masked(aligned_gt, aligned_pred, mask, data_range=1.0))

            if net_lpips or net_dists:
                pred_t = torch.from_numpy(aligned_pred).permute(2, 0, 1).unsqueeze(0).to(device)
                gt_t = torch.from_numpy(aligned_gt).permute(2, 0, 1).unsqueeze(0).to(device)
                if net_lpips:
                    lpips_list.append(net_lpips(pred_t * 2 - 1, gt_t * 2 - 1).item())
                if net_dists:
                    dists_list.append(net_dists(pred_t, gt_t).item())
        else:
            # GoPro / HIDE: RGB PSNR + 3D-Gaussian SSIM
            psnr_list.append(calc_psnr_rgb(pred, gt))
            ssim_list.append(calc_ssim_3d(pred, gt, device))

            if net_lpips or net_dists:
                pred_t = torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                gt_t = torch.from_numpy(gt).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                if net_lpips:
                    lpips_list.append(net_lpips(pred_t * 2 - 1, gt_t * 2 - 1).item())
                if net_dists:
                    dists_list.append(net_dists(pred_t, gt_t).item())

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "lpips": float(np.mean(lpips_list)) if lpips_list else None,
        "dists": float(np.mean(dists_list)) if dists_list else None,
        "n": len(psnr_list),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate deblur metrics from saved images")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gopro_orig", "hide", "realblur_j", "realblur_r"])
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--gopro_dir", type=str,
                        default="data/GOPRO_Large/test")
    parser.add_argument("--subframes_dir", type=str,
                        default="data/GOPRO_Large_subframes/test")
    parser.add_argument("--hide_dir", type=str,
                        default="data/HIDE_test")
    parser.add_argument("--realblur_dir", type=str,
                        default="data/RealBlur_test")
    parser.add_argument("--no_lpips", action="store_true")
    parser.add_argument("--no_dists", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Pred dir not found: {pred_dir}")

    use_ecc = False
    if args.dataset == "gopro_orig":
        pairs = load_gopro_orig_pairs(pred_dir, args.gopro_dir, args.subframes_dir)
    elif args.dataset == "hide":
        pairs = load_paired_dir(pred_dir, Path(args.hide_dir) / "target")
    elif args.dataset == "realblur_j":
        pairs = load_paired_dir(pred_dir, Path(args.realblur_dir) / "RealBlur-J" / "target")
        use_ecc = True
    elif args.dataset == "realblur_r":
        pairs = load_paired_dir(pred_dir, Path(args.realblur_dir) / "RealBlur-R" / "target")
        use_ecc = True

    print(f"Dataset:  {args.dataset}")
    print(f"Pred dir: {pred_dir}")
    print(f"Pairs:    {len(pairs)}")
    print(f"ECC:      {use_ecc}")
    print()

    results = evaluate(
        pairs,
        use_ecc=use_ecc,
        use_lpips=not args.no_lpips,
        use_dists=not args.no_dists,
    )

    label = f"{args.dataset} ({'ECC-aligned' if use_ecc else 'RGB'})"
    print("=" * 60)
    print(label)
    print("=" * 60)
    print(f"  Images: {results['n']}")
    print(f"  PSNR:   {results['psnr']:.2f} dB")
    print(f"  SSIM:   {results['ssim']:.4f}")
    if results['lpips'] is not None:
        print(f"  LPIPS:  {results['lpips']:.4f}")
    if results['dists'] is not None:
        print(f"  DISTS:  {results['dists']:.4f}")
    print("=" * 60)

    save_path = args.save or str(pred_dir.parent / f"metrics_{args.dataset}.txt")
    with open(save_path, 'w') as f:
        f.write(f"{label}\n")
        f.write(f"Images: {results['n']}\n")
        f.write(f"PSNR:   {results['psnr']:.2f} dB\n")
        f.write(f"SSIM:   {results['ssim']:.4f}\n")
        if results['lpips'] is not None:
            f.write(f"LPIPS:  {results['lpips']:.4f}\n")
        if results['dists'] is not None:
            f.write(f"DISTS:  {results['dists']:.4f}\n")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
