# Self-Describing Semantic Latent Platform

This folder is a compact sidecar platform for shape-metadata recovery experiments on top of the existing read-only `../SwinJSCC` implementation.

## Final Question

Can a receiver recover the tensor shape of a flattened semantic latent when a conventional metadata header fails? The final code compares three resource models:

- `Semantic Watermark`: zero extra side symbols, but injects distortion into the image latent.
- `Semantic Shape Header`: analog spread-spectrum semantic code for `(H, W, C)`, concatenated before the image latent.
- `Traditional Ultra-Redundant Header`: 3-bit shape ID protected by very-low-rate BPSK repetition or a random BPSK codebook with ML decoding.

The reviewer concern is explicitly tested: if the side-header baselines receive the same symbol budget as a semantic analog header, an ultra-redundant conventional code can remove the DVB-S2 threshold cliff and closely track the oracle.

## Main Scripts

```bash
source /root/anaconda3/bin/activate py310

# Optional: regenerate SwinJSCC watermark/header/original LDPC curves.
python run_full_psnr_curves.py --out-dir results/final_resource/base_regen --device cuda:0

# Optional: regenerate known-shape oracle data.
python run_oracle_upper_bound.py --out-dir results/final_resource/base_regen --device cuda:0

# Resource-matched comparison used for the final figures.
python run_resource_matched_comparison.py --trials 1000 --out-dir results/final_resource
```

## Final Outputs

All final CSV and figures are in `results/final_resource/`:

- `base_full_psnr_curves.csv`: cached SwinJSCC watermark, semantic-header, LDPC, and no-reference baseline data.
- `oracle_upper_bound.csv`: cached known-shape oracle PSNR data.
- `resource_matched_psnr.csv`: resource-matched metadata comparison.
- `resource_matched_psnr_overlay.png`: focused N=300 comparison.
- `resource_matched_psnr_by_budget.png`: N=96/300/768/1536 PSNR comparison.
- `resource_matched_shape_success.png`: metadata success curves.
- `resource_matched_summary.md`: short interpretation table.

## Key Interpretation

The watermark is not free: it trades latent fidelity for embedded shape information. If the system allows extra side symbols, a 300-symbol 3-bit traditional metadata code is already strong at very low SNR in this Kodak/SwinJSCC setting. The watermark should therefore be positioned as a zero-extra-side-symbol self-describing latent method, not as universally better than arbitrarily redundant channel coding.
