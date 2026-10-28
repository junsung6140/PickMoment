"""
PickMoment blur-to-video evaluation on GoPro-7 (Table 2), following the
Blur2Vid evaluation protocol.

Stages (all by default):
  1. infer: for each blur image under --blur_dir, extract --n_frames sharp
     frames at full resolution.
  2. save:  downsample each frame 2x (cv2.INTER_AREA, as in Blur2Vid) and write
         {output_dir}/preds/{scene}/{center_idx}/frame_00000.png ... frame_0000{N-1}.png
  3. eval:  compare against --gt_dir (640x360). Metrics are bidirectional
     (best of forward / reversed GT order, per metric):
         PSNR_p / SSIM_p / LPIPS_p : per-frame PSNR, 40x40 patch SSIM, 40x40 patch LPIPS(alex)
         PWPSNR, BlurryPSNR        : extra PSNR variants
         EPE (--with_epe)          : RAFT bidirectional EPE on (first, last) frames
         FVD (--with_fvd)          : cd-fvd with VideoMAE features
     Full-frame LPIPS / DISTS are also reported for reference.

Input blur layout (from scripts/prepare_gopro7_linear.py):
    {blur_dir}/{scene}/{center_idx}.png
GT layout (Blur2Vid GoPro-7 GT):
    {gt_dir}/{scene}/{center_idx}/frame_00000.png ... frame_0000{N-1}.png

Usage:
  python test_blur2vid.py --config configs/test.yaml \\
      --blur_dir data/GOPRO_7_linear/test/blur \\
      --gt_dir data/GOPROResultsImages/GT \\
      --output_dir results/blur2vid --with_epe --with_fvd

  # Evaluate existing predictions in {output_dir}/preds only
  python test_blur2vid.py --output_dir results/blur2vid --gt_dir ... --eval_only
"""

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim
from tqdm import tqdm


DEFAULT_GT_DIR = "data/GOPROResultsImages/GT"


# ============================================================
# Inference
# ============================================================

def pad_to_align(x, align_size=32):
    B, C, H, W = x.shape
    new_H = ((H + align_size - 1) // align_size) * align_size
    new_W = ((W + align_size - 1) // align_size) * align_size
    pad_h = new_H - H
    pad_w = new_W - W
    if pad_h == 0 and pad_w == 0:
        return x, (H, W)
    x_pad = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_pad, (H, W)


def unpad(x, hw):
    H, W = hw
    return x[..., :H, :W]


def load_blur_tensor(path, device):
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_uint8_2x_down(frame_tensor):
    """(1,3,H,W) or (3,H,W) in [-1,1] -> uint8 (H/2, W/2, 3) RGB via INTER_AREA."""
    if frame_tensor.dim() == 4:
        frame_tensor = frame_tensor[0]
    arr = ((frame_tensor + 1.0) / 2.0).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)
    arr_u8 = (arr * 255).clip(0, 255).round().astype(np.uint8)
    H, W = arr_u8.shape[:2]
    return cv2.resize(arr_u8, (W // 2, H // 2), interpolation=cv2.INTER_AREA)


def collect_blur_files(blur_dir):
    items = []
    for scene_dir in sorted(Path(blur_dir).iterdir()):
        if not scene_dir.is_dir():
            continue
        for p in sorted(scene_dir.glob("*.png")):
            items.append({"scene": scene_dir.name, "center_idx": p.stem, "path": str(p)})
    return items


@torch.no_grad()
def run_inference(model, blur_items, output_preds_dir, n_frames=7, align_size=32, eps=0.01):
    output_preds_dir = Path(output_preds_dir)
    device = next(model.transformer.parameters()).device

    for item in tqdm(blur_items, desc="Infer"):
        x_blur = load_blur_tensor(item["path"], device)
        x_pad, hw = pad_to_align(x_blur, align_size=align_size)
        frames = model.extract_video(x_pad, n_frames=n_frames, eps=eps)  # (1, N, 3, H, W)
        frames = unpad(frames, hw)

        clip_dir = output_preds_dir / item["scene"] / item["center_idx"]
        clip_dir.mkdir(parents=True, exist_ok=True)
        for fi in range(n_frames):
            arr_u8 = tensor_to_uint8_2x_down(frames[:, fi])
            Image.fromarray(arr_u8).save(clip_dir / f"frame_{fi:05d}.png")

        del x_blur, x_pad, frames


# ============================================================
# Blur2Vid evaluation protocol
# ============================================================

def read_frames_01(clip_dir):
    """Read all PNGs in clip_dir (sorted) -> (N, H, W, 3) float32 RGB in [0, 1]."""
    paths = sorted(glob.glob(os.path.join(clip_dir, "*.png")))
    frames = np.stack([cv2.imread(p, cv2.IMREAD_UNCHANGED)[..., ::-1] for p in paths])
    return np.clip(frames.astype(np.float32) / 255.0, 0, 1)


def collect_clip_pairs(pred_dir, gt_dir):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    pairs = []
    for scene in sorted(gt_dir.iterdir()):
        if not scene.is_dir():
            continue
        for cidx in sorted(scene.iterdir()):
            if not cidx.is_dir():
                continue
            clip = f"{scene.name}/{cidx.name}"
            pred_path = pred_dir / clip
            if pred_path.exists():
                pairs.append({"name": clip, "pred": str(pred_path), "gt": str(cidx)})
    return pairs


def psnr_per_frame(pred, gt):
    mse = ((pred - gt) ** 2).reshape(pred.shape[0], -1).mean(axis=1)
    return 10.0 * np.log10(1.0 / np.clip(mse, 1e-12, None))


def blurry_psnr(pred_frames, gt_frames):
    bp = np.mean(pred_frames ** 2.2, axis=0) ** (1 / 2.2)
    bg = np.mean(gt_frames ** 2.2, axis=0) ** (1 / 2.2)
    mse = np.mean((bp - bg) ** 2)
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def pw_psnr_bidir(pred_frames, gt_frames):
    fwd_mse = np.mean((pred_frames - gt_frames) ** 2, axis=0)
    bwd_mse = np.mean((pred_frames - gt_frames[::-1]) ** 2, axis=0)
    pw_best = np.minimum(fwd_mse, bwd_mse)
    return 10.0 * np.log10(1.0 / max(np.mean(pw_best), 1e-12))


def patch_ssim_lpips(pred_frames, gt_frames, lpips_net, device, patch_dim=40):
    """Mean SSIM and LPIPS over non-overlapping patch_dim x patch_dim patches."""
    N, H, W, _ = pred_frames.shape
    rows = [r for r in range(0, H, patch_dim) if r + patch_dim <= H]
    cols = [c for c in range(0, W, patch_dim) if c + patch_dim <= W]
    n_patches = len(rows) * len(cols)

    ssim_sum, lpips_sum, count = 0.0, 0.0, 0
    for i in range(N):
        pred = pred_frames[i]
        gt = gt_frames[i]
        for r0 in rows:
            for c0 in cols:
                ssim_sum += sk_ssim(
                    gt[r0:r0 + patch_dim, c0:c0 + patch_dim],
                    pred[r0:r0 + patch_dim, c0:c0 + patch_dim],
                    data_range=1.0, channel_axis=2,
                )
        pp = np.stack([pred[r:r + patch_dim, c:c + patch_dim] for r in rows for c in cols])
        gp = np.stack([gt[r:r + patch_dim, c:c + patch_dim] for r in rows for c in cols])
        pt = torch.from_numpy(pp).permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
        gt_t = torch.from_numpy(gp).permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
        with torch.no_grad():
            lpips_sum += float(lpips_net(pt, gt_t).view(-1).sum().item())
        count += n_patches
    return ssim_sum / count, lpips_sum / count


def full_frame_lpips_dists(pred_frames, gt_frames, lpips_net, dists_net, device):
    lp, di = [], []
    for i in range(pred_frames.shape[0]):
        pt = torch.from_numpy(pred_frames[i]).permute(2, 0, 1).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(gt_frames[i]).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            if lpips_net is not None:
                lp.append(lpips_net(pt * 2 - 1, gt_t * 2 - 1).view(-1).item())
            if dists_net is not None:
                di.append(dists_net(pt, gt_t).item())
    out = {}
    if lp:
        out["lpips_full"] = float(np.mean(lp))
    if di:
        out["dists_full"] = float(np.mean(di))
    return out


def compute_epe(pred_frames, gt_frames):
    """RAFT bidirectional EPE on the (first, last) frame pair."""
    from epe_metric import compute_bidirectional_epe
    val = compute_bidirectional_epe(
        torch.from_numpy(pred_frames[0]),
        torch.from_numpy(pred_frames[-1]),
        torch.from_numpy(gt_frames[0]),
        torch.from_numpy(gt_frames[-1]),
        per_pixel_mode=True,
    )
    return float(val.item())


def eval_clip(pred, gt, lpips_net, dists_net, device, patch_dim=40, full_frame=True,
              compute_epe_flag=False):
    score_blurry = blurry_psnr(pred, gt)

    fwd_psnr = float(np.mean(psnr_per_frame(pred, gt)))
    fwd_ssim, fwd_lpips = patch_ssim_lpips(pred, gt, lpips_net, device, patch_dim)
    bwd_psnr = float(np.mean(psnr_per_frame(pred, gt[::-1])))
    bwd_ssim, bwd_lpips = patch_ssim_lpips(pred, gt[::-1], lpips_net, device, patch_dim)

    out = {
        "psnr": max(fwd_psnr, bwd_psnr),
        "pwpsnr": pw_psnr_bidir(pred, gt),
        "blurry_psnr": score_blurry,
        "patch_ssim": max(fwd_ssim, bwd_ssim),
        "patch_lpips": min(fwd_lpips, bwd_lpips),
        "fwd_psnr": fwd_psnr,
        "bwd_psnr": bwd_psnr,
    }
    if full_frame and (lpips_net is not None or dists_net is not None):
        best_gt = gt if fwd_psnr >= bwd_psnr else gt[::-1]
        out.update(full_frame_lpips_dists(pred, best_gt, lpips_net, dists_net, device))

    if compute_epe_flag:
        out['epe'] = compute_epe(pred, gt)

    return out


def run_eval(preds_dir, gt_dir, patch_dim=40, no_dists=False, no_full=False,
             max_clips=None, do_epe=False, do_fvd=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = collect_clip_pairs(preds_dir, gt_dir)
    if max_clips is not None:
        pairs = pairs[:max_clips]
    if not pairs:
        raise RuntimeError(f"No matching clips between {preds_dir} and {gt_dir}")

    import lpips
    lpips_net = lpips.LPIPS(net="alex").to(device)
    lpips_net.requires_grad_(False)

    dists_net = None
    if not no_dists and not no_full:
        from DISTS_pytorch import DISTS
        dists_net = DISTS().to(device)
        dists_net.requires_grad_(False)

    fvd_engine = None
    if do_fvd:
        from cdfvd import fvd as _fvd_mod
        fvd_engine = _fvd_mod.cdfvd(model='videomae', device=device)

    print(f"Eval clips: {len(pairs)}  (patch={patch_dim}x{patch_dim}, epe={do_epe}, fvd={do_fvd})")

    agg = {}
    for p in tqdm(pairs, desc="Eval"):
        pred = read_frames_01(p["pred"])
        gt = read_frames_01(p["gt"])
        assert pred.shape == gt.shape, f"shape mismatch {p['name']}: {pred.shape} vs {gt.shape}"
        m = eval_clip(pred, gt, lpips_net, dists_net, device,
                      patch_dim=patch_dim, full_frame=not no_full,
                      compute_epe_flag=do_epe)
        for k, v in m.items():
            agg.setdefault(k, []).append(v)

        if fvd_engine is not None:
            gt_u8 = (gt * 255).astype(np.uint8)[None]
            pr_u8 = (pred * 255).astype(np.uint8)[None]
            fvd_engine.add_real_stats(gt_u8)
            fvd_engine.add_fake_stats(pr_u8)

    summary = {k: float(np.mean(v)) for k, v in agg.items()}

    if fvd_engine is not None:
        summary['fvd'] = float(fvd_engine.compute_fvd_from_stats())
        fvd_engine.empty_real_stats()
        fvd_engine.empty_fake_stats()

    return summary, len(pairs)


def format_results(summary, n):
    rows = [
        ("psnr", "PSNR_p       (up)"),
        ("patch_ssim", "SSIM_p       (up)"),
        ("patch_lpips", "LPIPS_p      (down)"),
        ("fvd", "FVD          (down)"),
        ("epe", "EPE          (down)"),
        ("pwpsnr", "PWPSNR       (up)"),
        ("blurry_psnr", "BlurryPSNR   (up)"),
        ("lpips_full", "LPIPS full   (down)"),
        ("dists_full", "DISTS full   (down)"),
    ]
    lines = ["=" * 60, f"GoPro-7 blur-to-video evaluation ({n} clips)", "=" * 60]
    for key, label in rows:
        if key in summary:
            lines.append(f"  {label} : {summary[key]:.4f}")
    if "fwd_psnr" in summary and "bwd_psnr" in summary:
        lines.append("-" * 60)
        lines.append(f"  fwd/bwd PSNR : {summary['fwd_psnr']:.2f} / {summary['bwd_psnr']:.2f}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="PickMoment GoPro-7 blur-to-video evaluation")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Predictions go to {output_dir}/preds, metrics to {output_dir}/metrics_blur2vid.txt")
    p.add_argument("--no_eval", action="store_true", help="Inference only")
    p.add_argument("--eval_only", action="store_true", help="Evaluate existing predictions only")
    # Inference
    p.add_argument("--config", type=str, default="configs/test.yaml",
                   help="Config with a 'model' section")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to best_model.pt. Overrides config")
    p.add_argument("--blur_dir", type=str, default="data/GOPRO_7_linear/test/blur",
                   help="{blur_dir}/{scene}/{center_idx}.png")
    p.add_argument("--n_frames", type=int, default=7)
    p.add_argument("--eps", type=float, default=0.01,
                   help="Half-width of the (s, t) query around each moment")
    p.add_argument("--align_size", type=int, default=32)
    # Eval
    p.add_argument("--gt_dir", type=str, default=DEFAULT_GT_DIR, help="GoPro-7 GT (640x360)")
    p.add_argument("--patch_dim", type=int, default=40)
    p.add_argument("--no_dists", action="store_true", help="Skip full-frame DISTS")
    p.add_argument("--no_full_frame", action="store_true", help="Skip full-frame LPIPS/DISTS")
    p.add_argument("--with_epe", action="store_true", help="Compute RAFT bidirectional EPE")
    p.add_argument("--with_fvd", action="store_true", help="Compute FVD (cd-fvd, VideoMAE)")
    p.add_argument("--max_clips", type=int, default=None, help="Limit clips for a quick check")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    preds_dir = output_dir / "preds"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        from models import build_model

        with open(args.config) as f:
            model_cfg = yaml.safe_load(f)["model"]
        if args.checkpoint:
            model_cfg["checkpoint"] = args.checkpoint
        if model_cfg.get("checkpoint") is None:
            raise ValueError("model.checkpoint missing (set in config or --checkpoint)")

        preds_dir.mkdir(parents=True, exist_ok=True)
        model = build_model(model_cfg, checkpoint=model_cfg["checkpoint"], mode="test")
        model.set_eval()

        blur_items = collect_blur_files(args.blur_dir)
        if args.max_clips is not None:
            blur_items = blur_items[:args.max_clips]
        print(f"Inference: {len(blur_items)} blur images -> {preds_dir}")
        run_inference(model, blur_items, preds_dir, n_frames=args.n_frames,
                      align_size=args.align_size, eps=args.eps)
        del model
        torch.cuda.empty_cache()

    if not args.no_eval:
        summary, n = run_eval(
            preds_dir, args.gt_dir,
            patch_dim=args.patch_dim,
            no_dists=args.no_dists,
            no_full=args.no_full_frame,
            max_clips=args.max_clips,
            do_epe=args.with_epe,
            do_fvd=args.with_fvd,
        )
        text = format_results(summary, n)
        print("\n" + text)

        metrics_path = output_dir / "metrics_blur2vid.txt"
        with open(metrics_path, "w") as f:
            f.write(text + "\n")
            f.write(f"preds_dir: {preds_dir}\n")
            f.write(f"gt_dir:    {args.gt_dir}\n")
            f.write(f"clips:     {n}\n")
        print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
