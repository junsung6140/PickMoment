"""
Build the GoPro-7 blur inputs for blur-to-video evaluation.

Blur is the direct average of 7 consecutive sub-frames in sRGB space (no gamma
conversion), which matches the blur synthesis used for PickMoment training.

For each N-frame window (default 7, non-overlapping stride):
  blur  = mean(sub-frames)        # linear sRGB average
  sharp = the same N frames copied to save_root/{split}/sharp/{scene}/

Usage:
  python scripts/prepare_gopro7_linear.py \\
      --src_root  data/GOPRO_Large_subframes \\
      --save_root data/GOPRO_7_linear \\
      --splits test --window_size 7
"""

import argparse
import os
import shutil
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm


def process_single_blur(task):
    seq_dir, img_files, start_idx, save_dir_blur, save_dir_sharp, window_size = task
    half = window_size // 2
    frame_indices = range(start_idx, start_idx + window_size)
    center_idx = start_idx + half

    scene_name = os.path.basename(seq_dir)
    blur_seq_dir = os.path.join(save_dir_blur, scene_name)
    sharp_seq_dir = os.path.join(save_dir_sharp, scene_name)
    os.makedirs(blur_seq_dir, exist_ok=True)
    os.makedirs(sharp_seq_dir, exist_ok=True)

    blur_img_path = os.path.join(blur_seq_dir, img_files[center_idx])
    if os.path.exists(blur_img_path):
        return

    # Linear sRGB average (no gamma correction)
    imgs = []
    for i in frame_indices:
        img = cv2.imread(os.path.join(seq_dir, img_files[i]), cv2.IMREAD_UNCHANGED)
        imgs.append(img.astype(np.float32) / 255.0)
    avg_img = np.mean(imgs, axis=0)
    avg_img = np.clip(avg_img, 0, 1)
    cv2.imwrite(blur_img_path, np.round(avg_img * 255).astype(np.uint8))

    # Copy the N sharp frames (source of this blur)
    for i in frame_indices:
        src = os.path.join(seq_dir, img_files[i])
        dst = os.path.join(sharp_seq_dir, img_files[i])
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)


def process_sequence(seq_dir, save_dir_blur, save_dir_sharp, window_size=7, workers=None):
    img_files = sorted(f for f in os.listdir(seq_dir) if f.endswith(".png"))
    tasks = []
    for w in range((len(img_files) - window_size) // window_size + 1):
        start_idx = w * window_size
        if start_idx + window_size <= len(img_files):
            tasks.append((seq_dir, img_files, start_idx,
                          save_dir_blur, save_dir_sharp, window_size))
    if not tasks:
        return
    with Pool(processes=workers or cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_single_blur, tasks),
                  total=len(tasks), desc=os.path.basename(seq_dir)))


def main():
    parser = argparse.ArgumentParser(description="Build GoPro-7 blur inputs (linear sRGB average)")
    parser.add_argument("--src_root", type=str,
                        default="data/GOPRO_Large_subframes")
    parser.add_argument("--save_root", type=str,
                        default="data/GOPRO_7_linear")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    for split in args.splits:
        split_src = os.path.join(args.src_root, split)
        if not os.path.isdir(split_src):
            print(f"Skip {split}: {split_src} not found")
            continue

        save_blur = os.path.join(args.save_root, split, "blur")
        save_sharp = os.path.join(args.save_root, split, "sharp")

        seq_dirs = sorted(
            os.path.join(split_src, d)
            for d in os.listdir(split_src)
            if os.path.isdir(os.path.join(split_src, d))
        )

        print(f"[{split}] {len(seq_dirs)} sequences -> {args.save_root}/{split}")
        for seq_dir in seq_dirs:
            process_sequence(seq_dir, save_blur, save_sharp,
                             args.window_size, args.workers)

    print(f"Done. GoPro-7 blur inputs at {args.save_root}")


if __name__ == "__main__":
    main()
