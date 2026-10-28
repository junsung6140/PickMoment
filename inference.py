"""
PickMoment inference: one blurry image (or a folder) in, sharp frames out.

Time inside the exposure is normalized to [-1, 1]: -1 is the start, 0 the
center, +1 the end.

Usage:
    # Deblur (center frame, tau = 0)
    python inference.py --input blurry.png --output_dir out

    # Pick moments: sharp frames at tau = -1, -0.5, 0.3
    python inference.py --input blurry.png --output_dir out --tau -1 -0.5 0.3

    # N-frame video (uniform tau from -1 to 1), also saved as mp4
    python inference.py --input blurry.png --output_dir out --n_frames 11 --save_mp4

    # Local checkpoint and backbone
    python inference.py --input blurry_dir/ --output_dir out \\
        --checkpoint checkpoints/best_model.pt --backbone checkpoints/FLUX.2-klein-4B
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm

from models import build_model

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
DEFAULT_CHECKPOINT = "checkpoints/best_model.pt"
DEFAULT_BACKBONE = "black-forest-labs/FLUX.2-klein-4B"


def pad_to_align(tensor, align_size=32):
    """Reflection-pad (B, C, H, W) so H and W are multiples of align_size."""
    _, _, h, w = tensor.shape
    pad_h = (align_size - h % align_size) % align_size
    pad_w = (align_size - w % align_size) % align_size
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return tensor, (pad_h, pad_w)


def unpad(tensor, pad_hw):
    pad_h, pad_w = pad_hw
    H, W = tensor.shape[-2], tensor.shape[-1]
    return tensor[..., :H - pad_h, :W - pad_w]


def load_image(path, align_size=32):
    img = Image.open(path).convert('RGB')
    tensor = (TF.to_tensor(img) * 2.0 - 1.0).unsqueeze(0)
    return pad_to_align(tensor, align_size)


def to_pil(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    return TF.to_pil_image(((tensor + 1.0) / 2.0).clamp(0, 1).float().cpu())


def collect_images(path):
    p = Path(path)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob('*') if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    return [str(p)]


def write_mp4(frames, out_path, fps=10):
    import cv2
    w, h = frames[0].size
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(np.array(f)[:, :, ::-1])
    writer.release()


@torch.no_grad()
def process_image(model, image_path, args):
    stem = Path(image_path).stem
    x_blur, pad_hw = load_image(image_path)
    x_blur = x_blur.to(model.device)

    if args.n_frames is None:
        taus = args.tau if args.tau else [0.0]
        for tau in taus:
            if not -1.0 <= tau <= 1.0:
                raise ValueError(f"tau must be in [-1, 1], got {tau}")
            pred = model.get_frame_at(x_blur, t_target=tau, eps=args.eps)
            name = f"{stem}_deblur.png" if (len(taus) == 1 and tau == 0.0) else f"{stem}_tau{tau:+.3f}.png"
            to_pil(unpad(pred, pad_hw)).save(os.path.join(args.output_dir, name))
    else:
        frames = model.extract_video(x_blur, n_frames=args.n_frames, eps=args.eps,
                                     max_batch=args.max_batch)
        frames = unpad(frames, pad_hw)
        video_dir = os.path.join(args.output_dir, stem)
        os.makedirs(video_dir, exist_ok=True)
        pil_frames = [to_pil(frames[:, fi]) for fi in range(args.n_frames)]
        for fi, img in enumerate(pil_frames):
            img.save(os.path.join(video_dir, f"frame_{fi:03d}.png"))
        if args.save_mp4:
            write_mp4(pil_frames, os.path.join(args.output_dir, f"{stem}.mp4"), fps=args.fps)


def parse_args():
    p = argparse.ArgumentParser(description="PickMoment inference")
    p.add_argument("--input", type=str, required=True, help="Image file or folder")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                   help="Path to best_model.pt")
    p.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                   help="FLUX.2-klein-4B folder or Hugging Face id (for the VAE and transformer config)")
    p.add_argument("--tau", type=float, nargs="+", default=None,
                   help="Moments in [-1, 1] to recover (default: 0, the center frame)")
    p.add_argument("--n_frames", type=int, default=None,
                   help="Extract an N-frame video at uniform moments from -1 to 1")
    p.add_argument("--eps", type=float, default=0.01,
                   help="Half-width of the (s, t) query around each moment")
    p.add_argument("--max_batch", type=int, default=None,
                   help="Max frames per forward pass for --n_frames (lower to save memory)")
    p.add_argument("--save_mp4", action="store_true")
    p.add_argument("--fps", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    if args.tau and args.n_frames:
        raise ValueError("Use either --tau or --n_frames, not both")

    image_files = collect_images(args.input)
    if not image_files:
        raise ValueError(f"No images found under {args.input}")
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model({'backbone': args.backbone}, checkpoint=args.checkpoint, mode='test')
    model.set_eval()

    for image_path in tqdm(image_files, desc="PickMoment"):
        process_image(model, image_path, args)
    print(f"Done -> {args.output_dir}")


if __name__ == "__main__":
    main()
