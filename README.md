<div align="center">

# PickMoment: Single-Image-to-Video via Learning Continuous-Time Reblurring

[Junseong Shin](https://junsung6140.github.io/)<sup>1</sup> · Hyeonsu Jo<sup>2</sup> · Daehyun Kim<sup>3</sup> · Tae Hyun Kim<sup>1,3†</sup>

<sup>1</sup>Department of Artificial Intelligence, Hanyang University · <sup>2</sup>Department of Intelligence Convergence, Hanyang University · <sup>3</sup>Department of Computer Science, Hanyang University

**NeurIPS 2026**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://junsung6140.github.io/pickmoment_project_page/)
[![arXiv](https://img.shields.io/badge/arXiv-2610.01279-b31b1b)](https://arxiv.org/abs/2610.01279)

</div>

![Overview](assets/teaser.jpg)

Overview of the proposed **PickMoment** framework. Given a single motion-blurred image, our method unifies three tasks: **(a)** Image deblurring, predicting the center sharp frame of the blur exposure. **(b)** Blur-to-video generation predicts an $n$-frame sharp video by uniformly sampling the normalized exposure trajectory at $M$ time points with interval $\Delta t = 1/(M-1)$; the figure illustrates an 11-frame example. **(c)** Arbitrary sharp-frame prediction enables the recovery of sharp frames at arbitrary time points within the blur exposure, thereby enabling continuous-time video generation.

## Abstract

Motion blur arises from the temporal integration of a continuous sharp signal over a finite exposure window, yet existing learning-based methods sidestep this physical model and predict only the sharp signal itself: most single-image deblurring methods recover a single frame at the exposure center, while blur-to-video methods predict a fixed set of frames. We introduce PickMoment, a continuous-time reformulation that directly learns the interval-mean blur over arbitrary sub-intervals of the exposure with a single deterministic model. Drawing an analogy to MeanFlow's average-velocity formulation, we train the model with three supervisions derived from the blur integral: an empirical reconstruction loss from available subframes, an additivity loss that enforces self-consistency across overlapping sub-intervals, and a sharp-frame loss anchored at the zero-interval limit. A single trained model unifies single-image deblurring, blur-to-video generation, and continuous-time pick-a-moment recovery as different queries to the same network, with no separate training for each task. Our PickMoment achieves state-of-the-art performance among generative-based deblurring methods on GoPro and HIDE while competitive against restoration-based methods on RealBlur, and the highest per-frame fidelity on GoPro-7 blur-to-video, all in a single forward pass without iterative sampling.

## Method

![Training strategy](assets/method.jpg)

Conceptual illustration of the training strategy of PickMoment.

## Results

Quantitative comparison for single-image deblurring. **Bold** indicates the best result and <u>underline</u> indicates the second-best result among generative-based methods.

| Dataset | Metric | Restormer | HI-Diff | UFPNet | AdaRevD | DiffBIR | OSEDiff | Diff-Plugin | FideDiff | Ours |
|---|---|---|---|---|---|---|---|---|---|---|
| GoPro | PSNR↑ | 32.92 | 33.33 | 34.09 | 34.60 | 26.15 | 24.34 | 22.88 | <u>28.79</u> | **29.43** |
| GoPro | SSIM↑ | 0.9611 | 0.9642 | 0.9686 | 0.9716 | 0.8377 | 0.8228 | 0.7798 | <u>0.9148</u> | **0.9253** |
| GoPro | LPIPS↓ | 0.0841 | 0.0799 | 0.0764 | 0.0712 | 0.2366 | 0.1738 | 0.2332 | <u>0.0831</u> | **0.0802** |
| GoPro | DISTS↓ | 0.0724 | 0.0710 | 0.0666 | 0.0672 | 0.1460 | 0.0834 | 0.1166 | <u>0.0525</u> | **0.0429** |
| HIDE | PSNR↑ | 31.22 | 31.46 | 31.74 | 32.35 | 25.94 | 23.20 | 21.94 | <u>27.28</u> | **27.52** |
| HIDE | SSIM↑ | 0.9423 | 0.9446 | 0.9471 | 0.9525 | 0.8216 | 0.7611 | 0.7272 | <u>0.8775</u> | **0.8891** |
| HIDE | LPIPS↓ | 0.1082 | 0.1055 | 0.0931 | 0.0872 | 0.2091 | 0.2208 | 0.2732 | <u>0.1068</u> | **0.1046** |
| HIDE | DISTS↓ | 0.0730 | 0.0725 | 0.0676 | 0.0666 | 0.1246 | 0.1001 | 0.1411 | <u>0.0647</u> | **0.0553** |
| RealBlur-J | PSNR↑ | 28.96 | 29.15 | 29.87 | 30.12 | 26.92 | 26.83 | 25.77 | **28.96** | <u>28.94</u> |
| RealBlur-J | SSIM↑ | 0.8786 | 0.8898 | 0.8840 | 0.8945 | 0.7450 | 0.8004 | 0.7711 | <u>0.8695</u> | **0.8757** |
| RealBlur-J | LPIPS↓ | 0.1561 | 0.1470 | 0.1441 | 0.1408 | 0.2587 | 0.1793 | 0.2055 | **0.1142** | <u>0.1184</u> |
| RealBlur-J | DISTS↓ | 0.1112 | 0.1050 | 0.1101 | 0.1037 | 0.1599 | 0.1198 | 0.1355 | **0.0800** | <u>0.0841</u> |
| RealBlur-R | PSNR↑ | 36.19 | 36.28 | 36.25 | 36.53 | 32.60 | 33.54 | 32.64 | <u>36.01</u> | **36.08** |
| RealBlur-R | SSIM↑ | 0.9572 | 0.9583 | 0.9528 | 0.9570 | 0.8493 | 0.9056 | 0.8496 | <u>0.9424</u> | **0.9510** |
| RealBlur-R | LPIPS↓ | 0.0608 | 0.0602 | 0.0615 | 0.0621 | 0.3388 | 0.1057 | 0.1422 | **0.0584** | <u>0.0612</u> |
| RealBlur-R | DISTS↓ | 0.0833 | 0.0814 | 0.0854 | 0.0846 | 0.2576 | 0.1322 | 0.1673 | <u>0.0862</u> | **0.0833** |

![Qualitative deblurring](assets/qual_deblur.jpg)

Qualitative comparison for single-image deblurring on GoPro, HIDE, and RealBlur-J (top to bottom). Best viewed zoomed in.

Quantitative comparison for blur-to-video on GoPro-7.

| Method | PSNR<sub>p</sub>↑ | SSIM<sub>p</sub>↑ | LPIPS<sub>p</sub>↓ | FVD↓ | EPE↓ |
|---|---|---|---|---|---|
| Jin et al. | 25.23 | 0.8190 | 0.084 | 235.53 | 3.38 |
| MotionETR | 26.54 | 0.8825 | 0.015 | 94.90 | 1.46 |
| Blur2Vid | <u>30.01</u> | <u>0.9359</u> | <u>0.012</u> | **21.46** | **0.39** |
| Ours | **31.24** | **0.9388** | **0.0086** | <u>27.40</u> | <u>0.86</u> |

![Qualitative blur-to-video](assets/qual_b2v.jpg)

Qualitative comparison for blur-to-video on GoPro-7. Best viewed zoomed in.

## Code

Coming soon.

## Citation

```bibtex
@inproceedings{shin2026pickmoment,
  title     = {PickMoment: Single-Image-to-Video via Learning Continuous-Time Reblurring},
  author    = {Shin, Junseong and Jo, Hyeonsu and Kim, Daehyun and Kim, Tae Hyun},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
