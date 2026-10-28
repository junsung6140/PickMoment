#!/bin/bash
# PickMoment full benchmark.
#   Deblurring (Tab. 1): GoPro / HIDE / RealBlur-J / RealBlur-R
#   Blur-to-video (Table 2): GoPro-7, Blur2Vid protocol (640x360, PSNR_p/SSIM_p/LPIPS_p, FVD, EPE)
#
# Usage:
#   bash scripts/test_all.sh
#   CHECKPOINT=checkpoints/best_model.pt OUT_DIR=results bash scripts/test_all.sh
set -e

CONFIG=${CONFIG:-configs/test.yaml}
OUT_DIR=${OUT_DIR:-results}
BLUR_DIR=${BLUR_DIR:-data/GOPRO_7_linear/test/blur}
GT_DIR=${GT_DIR:-data/GOPROResultsImages/GT}
PY=${PY:-python}

CKPT_ARG=""
[ -n "$CHECKPOINT" ] && CKPT_ARG="--checkpoint $CHECKPOINT"

echo "[1/5] GoPro deblurring"
$PY test.py --config "$CONFIG" $CKPT_ARG --mode gopro_deblur      --output_dir "$OUT_DIR/gopro"

echo "[2/5] HIDE deblurring"
$PY test.py --config "$CONFIG" $CKPT_ARG --mode hide_deblur       --output_dir "$OUT_DIR/hide"

echo "[3/5] RealBlur-J deblurring"
$PY test.py --config "$CONFIG" $CKPT_ARG --mode realblur_j_deblur --output_dir "$OUT_DIR/realblur_j"

echo "[4/5] RealBlur-R deblurring"
$PY test.py --config "$CONFIG" $CKPT_ARG --mode realblur_r_deblur --output_dir "$OUT_DIR/realblur_r"

echo "[5/5] GoPro-7 blur-to-video"
$PY test_blur2vid.py --config "$CONFIG" $CKPT_ARG \
    --blur_dir "$BLUR_DIR" \
    --gt_dir "$GT_DIR" \
    --output_dir "$OUT_DIR/blur2vid" \
    --n_frames 7 \
    --with_epe --with_fvd

echo "Done. Metrics:"
echo "  $OUT_DIR/gopro/metrics_gopro.txt"
echo "  $OUT_DIR/hide/metrics_hide.txt"
echo "  $OUT_DIR/realblur_j/metrics_realblur_j.txt"
echo "  $OUT_DIR/realblur_r/metrics_realblur_r.txt"
echo "  $OUT_DIR/blur2vid/metrics_blur2vid.txt"
