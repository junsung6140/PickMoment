"""
HIDE and RealBlur test sets.

Expected layout (both):
    {root}/input/*.png    blurry images
    {root}/target/*.png   GT sharp images (same file names)
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


def _to_tensor(img):
    return TF.to_tensor(img) * 2.0 - 1.0


class HIDETestDataset(Dataset):
    """HIDE test set. If `target/` is missing, GT is zeros and has_gt is False."""

    def __init__(self, hide_dir):
        hide_dir = Path(hide_dir)
        input_dir = hide_dir / "input"
        target_dir = hide_dir / "target"

        self.has_gt = target_dir.exists()
        self.samples = []
        for blur_path in sorted(input_dir.glob("*.png")):
            gt_path = target_dir / blur_path.name
            self.samples.append({
                "blur": str(blur_path),
                "gt": str(gt_path) if self.has_gt else None,
                "name": blur_path.stem,
            })

        print(f"HIDETestDataset: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        blur_img = Image.open(sample["blur"]).convert("RGB")
        gt_img = Image.open(sample["gt"]).convert("RGB") if self.has_gt else None

        x_blur = _to_tensor(blur_img)
        gt_sharp = _to_tensor(gt_img) if gt_img is not None else torch.zeros_like(x_blur)

        return {
            "x_blur": x_blur,
            "gt_sharp": gt_sharp,
            "filename": sample["name"],
            "has_gt": self.has_gt,
        }


class RealBlurTestDataset(Dataset):
    """RealBlur-J or RealBlur-R test set (pass the RealBlur-J or RealBlur-R folder)."""

    def __init__(self, realblur_dir):
        realblur_dir = Path(realblur_dir)
        input_dir = realblur_dir / "input"
        target_dir = realblur_dir / "target"

        self.samples = []
        for blur_path in sorted(input_dir.glob("*.png")):
            gt_path = target_dir / blur_path.name
            if gt_path.exists():
                self.samples.append({"blur": str(blur_path), "gt": str(gt_path),
                                     "name": blur_path.stem})

        print(f"RealBlurTestDataset ({realblur_dir.name}): {len(self.samples)} pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        blur_img = Image.open(sample["blur"]).convert("RGB")
        gt_img = Image.open(sample["gt"]).convert("RGB")
        return {
            "x_blur": _to_tensor(blur_img),
            "gt_sharp": _to_tensor(gt_img),
            "filename": sample["name"],
        }
