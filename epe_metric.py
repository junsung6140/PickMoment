"""
EPE (Endpoint Error) metric using RAFT optical flow.

Follows the Blur2Vid evaluation code (extra/epe_metric.py):
  - Input range [0, 1] (NOT [-1, 1]).
  - Resize to (H//8*8, W//8*8) instead of pad-to-8.
  - per_pixel_mode=True: pixel-wise min(EPE_fw, EPE_bw) then mean.
"""

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import resize as tv_resize


_raft_model = None


def _get_raft(device='cuda'):
    global _raft_model
    if _raft_model is None:
        weights = Raft_Large_Weights.DEFAULT
        _raft_model = raft_large(weights=weights).to(device).eval()
    return _raft_model


def _preprocess(img, device='cuda'):
    """Tensor (H, W, 3) [0,1] -> (1, 3, H', W') with H', W' multiples of 8."""
    if isinstance(img, torch.Tensor):
        t = img
    else:
        t = torch.from_numpy(img)
    t = t.float().to(device)
    if t.max() > 1.0:
        t = t / 255.0
    t = t.permute(2, 0, 1)               # (3, H, W)
    h, w = t.shape[1], t.shape[2]
    h, w = (h // 8) * 8, (w // 8) * 8
    t = tv_resize(t, [h, w])
    return t.unsqueeze(0)                # (1, 3, H, W)


def _estimate_flow(img1, img2, device='cuda'):
    model = _get_raft(device)
    t1 = _preprocess(img1, device)
    t2 = _preprocess(img2, device)
    with torch.no_grad():
        flow = model(t1, t2)[-1]         # (1, 2, H, W)
    return flow


def compute_bidirectional_epe(pred_first, pred_last, gt_first, gt_last,
                               per_pixel_mode=False):
    """Bidirectional EPE between predicted and GT frame pairs.

    pred_*, gt_*: (H, W, 3) float [0, 1] (or uint8). Convention matches
    blur2vid/extra/epe_metric.py.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    flow_gen_fw = _estimate_flow(pred_first, pred_last, device)
    flow_gen_bw = _estimate_flow(pred_last, pred_first, device)
    flow_gt = _estimate_flow(gt_first, gt_last, device)

    # Resize predicted flows to GT spatial size (no-op when shapes match)
    target_size = flow_gt.shape[-2:]
    flow_gen_fw = F.interpolate(flow_gen_fw, size=target_size, mode='bilinear', align_corners=False)
    flow_gen_bw = F.interpolate(flow_gen_bw, size=target_size, mode='bilinear', align_corners=False)

    # Per-pixel L2 norm of (pred_flow - gt_flow)
    epe_fw = torch.norm(flow_gen_fw.squeeze(0) - flow_gt.squeeze(0), dim=0)  # (H, W)
    epe_bw = torch.norm(flow_gen_bw.squeeze(0) - flow_gt.squeeze(0), dim=0)  # (H, W)

    if per_pixel_mode:
        # Pixel-wise best direction, then mean
        return torch.min(epe_fw, epe_bw).mean()
    else:
        # Direction-wise mean, then min
        return torch.tensor(min(epe_fw.mean().item(), epe_bw.mean().item()),
                            dtype=torch.float32)
