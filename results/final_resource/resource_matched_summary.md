# Resource-Matched Metadata Baselines

This experiment compares zero-side-bandwidth semantic watermarking against side-symbol metadata codes under equal side-symbol budgets.

- shape book: `[(32, 48, 96), (48, 32, 96)]`
- metadata bits for traditional code: `3`
- budgets: `[96, 300, 768, 1536]` real channel symbols
- trials for analog/ML header success: `1000` per shape/SNR/budget

Important interpretation: the semantic watermark spends no extra symbols but injects latent distortion. The side-header baselines spend explicit bandwidth and do not distort the image latent, so their utility equals oracle PSNR when the metadata decodes correctly.

## Focus: 300-symbol rate-1/100-style header

| SNR (dB) | Watermark PSNR | DVB-S2 LDPC PSNR | Analog Semantic Header PSNR | Traditional Repetition PSNR | Traditional ML Codebook PSNR |
|---:|---:|---:|---:|---:|---:|
| -12 | 9.65 | 0.00 | 8.67 | 9.30 | 9.46 |
| -10 | 10.88 | 0.00 | 10.54 | 10.88 | 10.90 |
| -8 | 13.17 | 0.00 | 13.05 | 13.10 | 13.11 |
| -6 | 15.74 | 0.00 | 16.28 | 16.28 | 16.28 |
| -4 | 19.23 | 0.00 | 19.35 | 19.35 | 19.35 |
| -2 | 22.67 | 23.00 | 22.34 | 22.34 | 22.34 |
| 0 | 25.98 | 25.64 | 26.05 | 26.05 | 26.05 |
| 2 | 28.17 | 28.24 | 28.28 | 28.28 | 28.28 |
| 4 | 29.77 | 29.80 | 29.81 | 29.81 | 29.81 |

## Conclusion

The reviewer concern is valid in a bandwidth-allowed setting: a very low-rate conventional side code can remove the LDPC cliff and closely track the oracle at the tested SNRs. The semantic watermark should therefore be positioned as a zero-extra-side-symbol self-describing latent mechanism, not as universally superior to arbitrarily redundant metadata channels.
