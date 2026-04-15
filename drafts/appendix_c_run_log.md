# Appendix C: Full Experiment Log

This appendix documents all deep learning training runs conducted on the school compute server (Kahan). Runs are grouped by experimental phase. Metrics are reported at the F2-optimal validation threshold unless otherwise noted. F2 is the primary metric; F1 is secondary. IoU is noted for completeness but is a poor standalone metric for this task without polygon dilation (see Section 5.7).

## Phase 1: Hyperparameter Exploration (v1 Mask, 256px Patches)

These initial runs established baseline performance and explored the hyperparameter space: backbone depth, input channels, loss composition, positive weight, and oversampling factor.

### DeepLabV3+ Runs

**Run 001** -- DeepLabV3+ / ResNet-18 / 10ch / batch 32 / lr 0.001 / pw=10 / 0.3 BCE + 0.7 Dice
- Best epoch 54/95. Val IoU 0.099, Test IoU 0.085.
- Initial baseline. Low performance with Dice-dominated loss.

**Run 002** -- DeepLabV3+ / ResNet-18 / 10ch / batch 32 / lr 0.001 / pw=10 / 0.5 BCE + 0.5 Dice
- Best epoch 27/68. Val IoU 0.090, Test IoU 0.075.
- Equal BCE/Dice weighting. Early stopping suggests limited signal in this configuration.

**Run 003** -- DeepLabV3+ / ResNet-18 / 10ch / batch 32 / lr 0.001 / pw=10 / 0.5 BCE + 0.5 Dice / 4x oversample
- Best epoch 35/76. Val IoU 0.103, Test IoU 0.117.
- Adding 4x oversampling improved test IoU by 56% over run 002.

**Run 004** -- DeepLabV3+ / ResNet-18 / 10ch / batch 32 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / cosine T_max=100
- Best epoch 46/87. Val IoU 0.105, Test IoU 0.116.
- First Lovasz run. Comparable to Dice-based run 003 but with cosine annealing.

**Run 005** -- DeepLabV3+ / ResNet-34 / 10ch / batch 32 / lr 0.001 / pw=50 / 0.5 BCE + 0.5 Lovasz / 8x oversample
- Best epoch 28/69. Val IoU 0.100, Test IoU 0.103.
- Higher pw=50 and 8x oversample did not improve over pw=30.

**Run 007** -- DeepLabV3+ / ResNet-34 / **4ch** / batch 32 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x oversample
- Best epoch 11/52. Val IoU 0.120, Test IoU 0.104.
- **Key finding: 4-channel input (RGBN only) outperforms 10-channel** (val IoU 0.120 vs. best 10ch 0.105). Models can learn spectral indices from raw bands.

**Run 009** -- DeepLabV3+ / ResNet-34 / 4ch / batch 16 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x / min_vacant=40
- Best epoch 50/91. Val IoU 0.117, Test IoU 0.110.
- Added min_vacant_pixels threshold. Marginal change.

**Run 010** -- DeepLabV3+ / ResNet-50 / 4ch / batch 16 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x / min_vacant=40
- Best epoch 39/80. Val IoU 0.115, Test IoU 0.106.
- ResNet-50 encoder. No improvement over ResNet-34 at this patch size.

**Run 011** -- DeepLabV3+ / ResNet-50 / 4ch / batch 16 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x / min_vacant=40 / band_dropout=0.0
- Best epoch 53/94. Val IoU 0.111, Test IoU 0.107.
- Band dropout disabled. Marginal difference from run 010.

**Run 016** -- DeepLabV3+ / ResNet-34 / 4ch / batch 32 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x
- Best epoch 35/76. Val IoU 0.095, Test IoU 0.107.
- ResNet-34 rerun; lower val IoU than run 007, suggesting some variance.

### UNet Runs

**Run 006** -- UNet / ResNet-18 / 10ch / batch 32 / lr 0.001 / pw=1.0 / pure Dice
- Best epoch 39/60. Val IoU 0.089, Test IoU 0.097.
- Pure Dice loss baseline. Lowest performance.

**Run 007** -- UNet / ResNet-18 / 10ch / batch 12 / lr 0.001 / pw=10 / 0.3 BCE + 0.7 Dice
- Best epoch 39/80. Val IoU 0.103, Test IoU 0.092.
- BCE + Dice combination improves over pure Dice.

**Run 009** -- UNet / ResNet-34 / 10ch / batch 12 / lr 0.001 / pw=10 / 0.5 BCE + 0.5 Dice
- Best epoch 59/100. Val IoU 0.099, Test IoU 0.083.
- ResNet-34 with 10ch; performance comparable to ResNet-18.

**Run 010** -- UNet / ResNet-34 / 10ch / batch 12 / lr 0.001 / pw=10 / 0.5 BCE + 0.5 Dice / 4x oversample
- Best epoch 52/93. Val IoU 0.117, Test IoU 0.127.
- 4x oversampling provides significant boost (+18% val IoU vs. run 009).

**Run 011** -- UNet / ResNet-34 / 10ch / batch 12 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / cosine T_max=100
- Best epoch 86/127. Val IoU 0.106, Test IoU 0.125.
- Lovasz + cosine schedule. Longer training needed.

**Run 012** -- UNet / ResNet-34 / 10ch / batch 12 / lr 0.001 / pw=50 / 0.5 BCE + 0.5 Lovasz / 8x oversample
- Best epoch 36/77. Val IoU 0.113, Test IoU 0.107.
- Higher pw=50 and 8x oversample.

**Run 013** -- UNet / ResNet-34 / **4ch** / batch 12 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x
- Best epoch 35/76. Val IoU 0.129, Test IoU 0.131.
- **Best Phase 1 result.** 4ch outperforms 10ch again (val IoU 0.129 vs best 10ch 0.117).

**Run 014** -- UNet / ResNet-34 / 4ch / batch 4 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x / min_vacant=40
- Best epoch 1/42. Val IoU 0.039, Test IoU 0.032.
- **Collapsed.** Batch size 4 too small; degenerate solution (predicts everything vacant, recall ~1.0, precision ~3%).

## Phase 2: Scaling Up (v1 Mask, 512/1024px Patches)

Increased patch size to provide broader spatial context.

### DeepLabV3+

**Run 015** -- DeepLabV3+ / ResNet-50 / 4ch / **1024px** / batch 8 / lr 0.0005 / pw=30 / 0.5 BCE + 0.5 Lovasz / 4x / min_vacant=1000 / band_dropout=0.3 / eval_stride=512
- Best epoch 94/135. Val IoU 0.125, Test IoU 0.136.
- Scale-up to 1024px. Test IoU improves over best 256px runs. AP improves to 0.137 (val), 0.159 (test).

### UNet

**Run 015** -- UNet / ResNet-50 / 4ch / **512px** / batch 8 / lr 0.0005 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x / min_vacant=40
- Best epoch 50/91. Val IoU 0.144, Test IoU 0.128.
- Scale-up to 512px. Best UNet result to date.

**Run 017** -- UNet / ResNet-50 / 4ch / 256px / batch 8 / lr 0.001 / pw=30 / 0.5 BCE + 0.5 Lovasz / 8x / min_vacant=40 / band_dropout=0.0
- Best epoch 37/78. Val IoU 0.127, Test IoU 0.125.
- 256px with band dropout disabled. Confirms 512px superiority.

**Run 027** -- UNet / ResNet-18 / 4ch / 1024px / batch 2 / lr 0.0005 / pw=30 / 0.5 BCE + 0.5 Lovasz / 4x / min_vacant=1000 / band_dropout=0.3
- Best epoch 1/42. Val IoU 0.046, Test IoU 0.039.
- **Collapsed.** ResNet-18 too shallow for 1024px patches with batch 2.

## Phase 3: v2 Mask and Training Stabilization

Introduced corrected vacancy mask (v2), building probability channel, linear warmup, and gradient clipping.

### DeepLabV3+

**Run 018** -- DeepLabV3+ / ResNet-50 / **5ch** (RGBN + building) / 1024px / batch 8 / lr 0.0005 / v2 mask
- Best epoch 76/117. Val F2 **0.405**, Test F2 **0.385**. Val IoU 0.154, Test IoU 0.149.
- First v2 mask run. Building channel added. **Best F2 of any run.** High-recall regime: val recall 0.617, val precision 0.170.

**Run 019** -- identical to 018
- Best epoch 2/43. Val IoU 0.088, Test IoU 0.069.
- **Collapsed at epoch 2.** CUDA non-determinism + narrow convergence basin.

**Run 020** -- identical to 018
- Best epoch 2/43. Val IoU 0.081, Test IoU 0.054.
- **Collapsed at epoch 2.** Second consecutive collapse motivates warmup/grad clip.

**Run 024** -- DeepLabV3+ / ResNet-50 / 5ch / 1024px / batch 8 / lr 0.0005 / v2 mask / **warmup 5ep** / **grad_clip=1.0**
- Best epoch 58/99. Val F2 0.295, Test F2 0.197. Val IoU **0.193**, Test IoU 0.134.
- Warmup + grad clipping resolve collapse instability and achieve best val IoU of any run. However, warmup shifts the model toward a precision-oriented decision boundary: val precision 0.388, val recall 0.278. Lower F2 than run 018 (no warmup) despite higher IoU.

### UNet

**Run 028** -- UNet / ResNet-34 / **5ch** / 512px / batch 4 / lr 0.0005 / v1 mask
- Best epoch 216/257. Val IoU 0.117, Test IoU 0.150.
- 5ch on v1 mask. Very long training (216 epochs).

**Run 029** -- UNet / ResNet-34 / 5ch / 512px / batch 4 / lr 0.0005 / **v2 mask**
- Best epoch 131/172. Val F2 **0.385**, Test F2 0.328. Val IoU 0.143, Test IoU 0.122.
- v2 mask with building channel. **Best UNet val F2.** High recall: val 0.602, test 0.497.

**Run 030** -- UNet / ResNet-34 / 5ch / 512px / batch 4 / lr 0.0005 / v2 mask
- Best epoch 92/133. Val F2 0.367, Test F2 **0.341**. Val IoU 0.135, Test IoU 0.125.
- **Best UNet test F2.** Similar recall-oriented regime to run 029: val recall 0.580, test recall 0.530.

**Run 031** -- UNet / ResNet-34 / **4ch** / 512px / batch 4 / lr 0.0005 / v2 mask
- Best epoch 46/87. Val F2 0.347, Test F2 0.277. Val IoU **0.169**, Test IoU **0.141**.
- Best UNet IoU. 4ch is more precision-oriented than 5ch (val prec 0.227, val recall 0.400). Beats 029/030 on IoU but falls behind on F2.

## Phase 4: Additional Runs

[PLACEHOLDER: If F2-priority runs complete, add here.]

## Summary Table

**Table C.1: Top runs by test F2** (primary metric)

| Rank | Run | Arch | Encoder | Ch | Patch | Mask | Val F2 | Test F2 | Test F1 | Test Recall | Test IoU |
|:----:|:---:|------|---------|:--:|:-----:|:----:|-------:|--------:|--------:|------------:|---------:|
| 1 | 018 | DLV3+ | RN50 | 5 | 1024 | v2 | 0.405 | 0.385 | 0.259 | 0.570 | 0.149 |
| 2 | 030 | UNet | RN34 | 5 | 512 | v2 | 0.367 | 0.341 | 0.222 | 0.530 | 0.125 |
| 3 | 029 | UNet | RN34 | 5 | 512 | v2 | 0.385 | 0.328 | 0.218 | 0.497 | 0.122 |
| 4 | 013 | UNet | RN34 | 4 | 256 | v1 | 0.293 | 0.330 | 0.231 | 0.461 | 0.131 |
| 5 | 031 | UNet | RN34 | 4 | 512 | v2 | 0.347 | 0.277 | 0.248 | 0.301 | 0.141 |

Note: Test IoU is included for reference but is not the sorting metric. Run 031 ranks higher by test IoU (0.141) than runs 029 and 030 (0.122, 0.125), illustrating how IoU and F2 can produce different rankings for models with different precision-recall operating points.
