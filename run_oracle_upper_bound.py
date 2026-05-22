"""
Compute oracle upper bound: SwinJSCC PSNR when receiver knows the correct
tensor shape (no watermark, no header, no shape ambiguity — ideal case).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sdlatent.channel import awgn
from sdlatent.swin_codec import build_swin, decode_latent, load_image_tensor, psnr01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swin-dir", default="../SwinJSCC")
    parser.add_argument("--image-dir", default="../SwinJSCC/Kodak_dataset")
    parser.add_argument("--model-path", default="../SwinJSCC/models/SwinJSCC_w_SA_AWGN_HRimage_snr_psnr_C96.model")
    parser.add_argument("--model", default="SwinJSCC_w/_SA")
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--c", type=int, default=96)
    parser.add_argument("--encoder-snr", type=float, default=10.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--images", default="kodim01,kodim02,kodim05,kodim08,kodim13,kodim15,kodim19,kodim20,kodim23,kodim24")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--out-dir", default="results/full_curves")
    args = parser.parse_args()

    image_names = [n.strip() for n in args.images.split(",")]
    snrs = np.arange(-12.0, 4.5, 0.5)
    rng = np.random.default_rng(args.seed)
    net, device = build_swin(args)
    out_dir = Path(args.out_dir)

    import torch

    oracle_rows = []
    for img_name in image_names:
        img_path = Path(args.image_dir) / f"{img_name}.png"
        tensor, hw = load_image_tensor(img_path, device)
        original_np = tensor.cpu().numpy()[0]
        actual_shape = (hw[0] // 16, hw[1] // 16, args.c)

        with torch.inference_mode():
            net.encoder.update_resolution(*hw)
            feature = net.encoder(tensor, args.encoder_snr, args.c, args.model)
            if isinstance(feature, tuple):
                feature = feature[0]
        latent = feature.cpu().float().numpy().reshape(-1)

        for snr_db in snrs:
            noise_rng = np.random.default_rng(args.seed + int((snr_db + 12) * 100))
            rx = awgn(latent, snr_db, noise_rng)
            recon = decode_latent(net, rx, actual_shape, snr_db, device, args.model)
            psnr_val = psnr01(original_np, recon)
            oracle_rows.append({
                "image": img_name, "snr_db": snr_db,
                "oracle_psnr": psnr_val,
            })
        print(f"  {img_name}: oracle -12dB={oracle_rows[-33]['oracle_psnr']:.1f}  0dB={oracle_rows[-10]['oracle_psnr']:.1f}  +4dB={oracle_rows[-1]['oracle_psnr']:.1f}")

    # Save oracle CSV
    oracle_path = out_dir / "oracle_upper_bound.csv"
    with oracle_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "snr_db", "oracle_psnr"])
        writer.writeheader()
        writer.writerows(oracle_rows)

    # Aggregate oracle PSNR
    oracle_agg = {}
    for snr_db in snrs:
        vals = [r["oracle_psnr"] for r in oracle_rows if r["snr_db"] == snr_db]
        oracle_agg[snr_db] = float(np.mean(vals))

    # Save aggregate
    agg_path = out_dir / "oracle_aggregate.csv"
    with agg_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "oracle_psnr_mean"])
        writer.writeheader()
        for snr_db in snrs:
            writer.writerow({"snr_db": snr_db, "oracle_psnr_mean": oracle_agg[snr_db]})

    print(f"\nSaved {oracle_path} and {agg_path}")
    print(f"Oracle PSNR: -12dB={oracle_agg[-12.0]:.1f}  -8dB={oracle_agg[-8.0]:.1f}  -4dB={oracle_agg[-4.0]:.1f}  0dB={oracle_agg[0.0]:.1f}  +4dB={oracle_agg[4.0]:.1f}")


if __name__ == "__main__":
    main()
