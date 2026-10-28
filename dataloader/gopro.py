"""
GoPro datasets built from the raw 240 fps sub-frames.

Expected layout:
    GOPRO_Large/{split}/{scene}/"frames N offset M.txt"   (only used to read N per scene)
    GOPRO_Large_subframes/{split}/{scene}/*.png           (all raw 240 fps frames)

For each scene with N averaged frames per blur, the k-th blur is the pixel
average of sub-frames [k*N, (k+1)*N), and its GT sharp frame is the center one.
Time inside the exposure is normalized so that sub-frame 0 is -1 and
sub-frame N-1 is +1.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# --------------------------------------------------------------------------
# Augmentation (8-fold flip + rotation, paired random crop)
# --------------------------------------------------------------------------

def paired_random_crop(imgs, patch_size):
    """Random crop a list of PIL images at the same location."""
    w, h = imgs[0].size

    if w < patch_size or h < patch_size:
        scale = max(patch_size / w, patch_size / h) * 1.01
        new_w, new_h = int(w * scale), int(h * scale)
        new_w = new_w + (new_w % 2)
        new_h = new_h + (new_h % 2)
        imgs = [img.resize((new_w, new_h), Image.BICUBIC) for img in imgs]
        w, h = new_w, new_h

    left = random.randint(0, w - patch_size)
    top = random.randint(0, h - patch_size)
    return [TF.crop(img, top, left, patch_size, patch_size) for img in imgs]


def paired_augment(imgs, mode=None):
    """Apply one of 8 geometric transforms (flip / rot90) to all images."""
    if mode is None:
        mode = random.randint(0, 7)

    results = []
    for img in imgs:
        if mode == 0:
            out = img
        elif mode == 1:
            out = TF.vflip(img)
        elif mode == 2:
            out = TF.rotate(img, 90, expand=True)
        elif mode == 3:
            out = TF.vflip(TF.rotate(img, 90, expand=True))
        elif mode == 4:
            out = TF.rotate(img, 180)
        elif mode == 5:
            out = TF.vflip(TF.rotate(img, 180))
        elif mode == 6:
            out = TF.rotate(img, 270, expand=True)
        elif mode == 7:
            out = TF.vflip(TF.rotate(img, 270, expand=True))
        results.append(out)

    return results


# --------------------------------------------------------------------------
# Scene parsing
# --------------------------------------------------------------------------

def _parse_n_orig(scene_dir):
    """Number of averaged frames per blur for a GOPRO_Large scene."""
    for txt in scene_dir.glob("frames *.txt"):
        parts = txt.stem.split()
        if len(parts) >= 2:
            return int(parts[1])
    # Fallback: parse from folder name (e.g. GOPR0384_11_00 -> 11)
    for part in scene_dir.name.split("_"):
        if part.isdigit() and int(part) in (7, 9, 11, 13, 15):
            return int(part)
    return None


def _collect_scenes(gopro_dir, subframes_dir, filter_n=None):
    """Return {scene_name: [window dicts]} in sorted scene order."""
    gopro_dir = Path(gopro_dir)
    subframes_dir = Path(subframes_dir)

    scene_samples = {}
    for scene_dir in sorted(gopro_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name
        sf_scene = subframes_dir / scene_name
        if not sf_scene.exists():
            continue

        n_orig = _parse_n_orig(scene_dir)
        if n_orig is None:
            continue
        if filter_n is not None and n_orig != filter_n:
            continue

        sf_paths = sorted(sf_scene.glob("*.png"))
        n_blur = len(sf_paths) // n_orig

        scene_list = []
        for blur_idx in range(n_blur):
            center = blur_idx * n_orig + n_orig // 2
            half = n_orig // 2
            if center - half < 0 or center + half >= len(sf_paths):
                continue
            scene_list.append({
                "sf_paths": [str(p) for p in sf_paths],
                "center": center,
                "n_orig": n_orig,
                "name": f"{scene_name}_seq{blur_idx:04d}",
            })
        scene_samples[scene_name] = scene_list
    return scene_samples


def _subsample_per_scene(scene_samples, max_per_scene):
    samples = []
    for _, scene_list in scene_samples.items():
        if max_per_scene is not None and len(scene_list) > max_per_scene:
            step = len(scene_list) / max_per_scene
            selected = [scene_list[int(i * step)] for i in range(max_per_scene)]
        else:
            selected = scene_list
        samples.extend(selected)
    return samples


def _load_window(sample):
    n = sample["n_orig"]
    half = n // 2
    center = sample["center"]
    paths = sample["sf_paths"]
    return np.stack([
        np.array(Image.open(paths[center - half + i]).convert("RGB"))
        for i in range(n)
    ])  # (n, H, W, 3) uint8


# --------------------------------------------------------------------------
# Training dataset
# --------------------------------------------------------------------------

class GoProTrainDataset(Dataset):
    """
    Random (s, t) interval samples from GoPro training scenes.

    Each item:
        x_blur        : pixel average of all N sub-frames, [-1, 1]
        gt_avg        : mean of the GT sub-frames in [s, t], [-1, 1]
        gt_boundary_s : sub-frame at s, [-1, 1]
        gt_boundary_t : sub-frame at t, [-1, 1]
        s, t          : normalized times in [-1, 1] (s == t for sharp samples)

    frame_parity:
        'all'  : every sub-frame index can be an endpoint or inside gt_avg.
        'odd'  : only odd indices (1, 3, 5, ...) are used as endpoints and in
                 gt_avg; even indices are never seen as targets.
        'even' : the reverse.
    """

    def __init__(self, gopro_dir, subframes_dir, patch_size=512,
                 epoch_size=50000, sharp_ratio=0.2, frame_parity='all'):
        self.patch_size = patch_size
        self.epoch_size = epoch_size
        self.sharp_ratio = sharp_ratio
        assert frame_parity in ('all', 'even', 'odd'), frame_parity
        self.frame_parity = frame_parity

        scene_samples = _collect_scenes(gopro_dir, subframes_dir)
        self.samples = [s for scene_list in scene_samples.values() for s in scene_list]
        if not self.samples:
            raise RuntimeError(f"No samples found in {gopro_dir}")

        print(f"GoProTrainDataset: {len(self.samples)} samples "
              f"(patch={patch_size}, epoch={epoch_size}, "
              f"sharp_ratio={sharp_ratio}, frame_parity={frame_parity})")

    def __len__(self):
        return self.epoch_size

    def _allowed_pool(self, n_frames):
        if self.frame_parity == 'even':
            return list(range(0, n_frames, 2))
        if self.frame_parity == 'odd':
            return list(range(1, n_frames, 2))
        return list(range(n_frames))

    def _sample_interval(self, n_frames):
        if self.frame_parity != 'all':
            return self._sample_interval_parity(n_frames)
        if random.random() < self.sharp_ratio:
            s_idx = random.randint(0, n_frames - 1)
            t_idx = s_idx
        else:
            s_idx = random.randint(0, n_frames - 2)
            t_idx = random.randint(s_idx + 1, n_frames - 1)
        return s_idx, t_idx

    def _sample_interval_parity(self, n_frames):
        """Pick (s_idx, t_idx) from the allowed parity pool (sorted)."""
        pool = self._allowed_pool(n_frames)
        if len(pool) == 0:
            return 0, 0
        if len(pool) == 1 or random.random() < self.sharp_ratio:
            i = random.choice(pool)
            return i, i
        a, b = random.sample(pool, 2)
        return (a, b) if a < b else (b, a)

    @staticmethod
    def _normalize_time(frame_idx, n_frames):
        mid = (n_frames - 1) / 2.0
        if mid == 0:
            return 0.0
        return (frame_idx - mid) / mid

    def __getitem__(self, _):
        sample = random.choice(self.samples)
        n = sample["n_orig"]
        window = _load_window(sample)

        s_idx, t_idx = self._sample_interval(n)

        pil_frames = [Image.fromarray(window[i]) for i in range(n)]

        # Blur = pixel average of all n sub-frames
        blur_np = window.astype(np.float32).mean(axis=0).round().astype(np.uint8)
        blur_pil = Image.fromarray(blur_np)

        all_imgs = [blur_pil] + pil_frames
        all_imgs = paired_random_crop(all_imgs, self.patch_size)
        all_imgs = paired_augment(all_imgs)

        blur_pil = all_imgs[0]
        pil_frames = all_imgs[1:]

        # GT average over allowed frames in [s_idx, t_idx]
        if self.frame_parity == 'all':
            avg_indices = list(range(s_idx, t_idx + 1))
        else:
            avg_indices = [i for i in self._allowed_pool(n) if s_idx <= i <= t_idx]
            if not avg_indices:
                avg_indices = [s_idx]
        interval_tensors = torch.stack([TF.to_tensor(pil_frames[i]) for i in avg_indices])
        gt_avg_t = interval_tensors.mean(dim=0)

        boundary_s_t = TF.to_tensor(pil_frames[s_idx])
        boundary_t_t = TF.to_tensor(pil_frames[t_idx])

        s_norm = self._normalize_time(s_idx, n)
        t_norm = self._normalize_time(t_idx, n)

        blur_t = TF.to_tensor(blur_pil) * 2.0 - 1.0
        gt_avg_t = gt_avg_t * 2.0 - 1.0
        boundary_s_t = boundary_s_t * 2.0 - 1.0
        boundary_t_t = boundary_t_t * 2.0 - 1.0

        return {
            "x_blur": blur_t,
            "gt_avg": gt_avg_t,
            "gt_boundary_s": boundary_s_t,
            "gt_boundary_t": boundary_t_t,
            "s": torch.tensor(s_norm, dtype=torch.float32),
            "t": torch.tensor(t_norm, dtype=torch.float32),
            "n_frames": n,
            "filename": sample["name"],
        }


# --------------------------------------------------------------------------
# Validation dataset (used during training)
# --------------------------------------------------------------------------

class GoProValDataset(Dataset):
    """
    Blur + all N GT sub-frames, center-cropped to a multiple of `align_size`.
    Used for validation during training.
    """

    def __init__(self, gopro_dir, subframes_dir, align_size=32, max_per_scene=None):
        self.align_size = align_size
        scene_samples = _collect_scenes(gopro_dir, subframes_dir)
        self.samples = _subsample_per_scene(scene_samples, max_per_scene)

        total_full = sum(len(v) for v in scene_samples.values())
        print(f"GoProValDataset: {len(self.samples)} items "
              f"(from {total_full} total, {len(scene_samples)} scenes"
              f"{f', max {max_per_scene}/scene' if max_per_scene else ''})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        n = sample["n_orig"]
        frames_np = _load_window(sample)

        H, W = frames_np.shape[1], frames_np.shape[2]
        new_H = (H // self.align_size) * self.align_size
        new_W = (W // self.align_size) * self.align_size

        def to_tensor(arr_hwc):
            t = TF.to_tensor(Image.fromarray(arr_hwc))
            t = t * 2.0 - 1.0
            if new_H != H or new_W != W:
                t = TF.center_crop(t, (new_H, new_W))
            return t

        x_blur_np = frames_np.astype(np.float32).mean(axis=0).round().astype(np.uint8)
        x_blur_t = to_tensor(x_blur_np)

        gt_frames = torch.stack([to_tensor(frames_np[i]) for i in range(n)])
        gt_sharp = gt_frames[n // 2]

        return {
            "x_blur": x_blur_t,
            "gt_sharp": gt_sharp,
            "gt_frames": gt_frames,
            "n_frames": n,
            "filename": sample["name"],
        }


# --------------------------------------------------------------------------
# Test dataset (full resolution, used by test.py)
# --------------------------------------------------------------------------

class GoProTestDataset(Dataset):
    """
    GoPro test set at full resolution (no crop; test.py pads to 32).

    Returns:
        x_blur    : (3, H, W) in [-1, 1], pixel average of the N sub-frames
        gt_sharp  : (3, H, W) in [-1, 1], center sub-frame
        gt_frames : (N, 3, H, W) in [-1, 1]
        n_frames  : N
        filename  : "{scene}_seq{k:04d}"

    filter_n: keep only scenes with N == filter_n (e.g. 11 for GoPro-11).
    """

    def __init__(self, gopro_dir, subframes_dir, max_per_scene=None, filter_n=None):
        self.filter_n = filter_n
        scene_samples = _collect_scenes(gopro_dir, subframes_dir, filter_n=filter_n)
        self.samples = _subsample_per_scene(scene_samples, max_per_scene)

        total_full = sum(len(v) for v in scene_samples.values())
        suffix = ""
        if filter_n is not None:
            suffix += f", filter_n={filter_n}"
        if max_per_scene is not None:
            suffix += f", max {max_per_scene}/scene"
        print(f"GoProTestDataset: {len(self.samples)} items "
              f"(from {total_full} total, {len(scene_samples)} scenes{suffix})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        n = sample["n_orig"]
        frames_np = _load_window(sample)

        def to_tensor(arr_hwc):
            return TF.to_tensor(Image.fromarray(arr_hwc)) * 2.0 - 1.0

        x_blur_np = frames_np.astype(np.float32).mean(axis=0).round().astype(np.uint8)
        x_blur_t = to_tensor(x_blur_np)

        gt_frames = torch.stack([to_tensor(frames_np[i]) for i in range(n)])
        gt_sharp = gt_frames[n // 2]

        return {
            "x_blur": x_blur_t,
            "gt_sharp": gt_sharp,
            "gt_frames": gt_frames,
            "n_frames": n,
            "filename": sample["name"],
        }
