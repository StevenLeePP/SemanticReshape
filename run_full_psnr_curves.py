"""
PSNR-vs-SNR curves for 10 Kodak images, 0.5 dB steps, all methods.
Also generates per-image reconstruction grids at key SNRs.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdlatent.baselines import choose_shape_by_smoothness, latent_smoothness_score
from sdlatent.channel import awgn
from sdlatent.reference_ldpc import PublishedThresholdLDPC
from sdlatent.semantic_header import SemanticShapeHeader
from sdlatent.shape_code import RobustSpreadSpectrumShapeComponents, SpreadSpectrumShapeComponents
from sdlatent.swin_codec import build_swin, decode_latent, load_image_tensor, psnr01


def decode_and_reencode(net, noisy_latent, shape, snr_db, device, model, c):
    import torch
    try:
        recon = decode_latent(net, noisy_latent, shape, snr_db, device, model)
    except Exception:
        return None
    h, w, _ = shape
    img_h, img_w = h * 16, w * 16
    tensor = torch.from_numpy(recon).float().unsqueeze(0).to(device)
    net.encoder.update_resolution(img_h, img_w)
    with torch.inference_mode():
        feature = net.encoder(tensor, snr_db, c, model)
        if isinstance(feature, tuple):
            feature = feature[0]
    return feature.cpu().float().numpy().reshape(-1)


def reencode_distance(net, noisy_latent, shape, snr_db, device, model, c):
    reencoded = decode_and_reencode(net, noisy_latent, shape, snr_db, device, model, c)
    if reencoded is None:
        return float("inf")
    return float(np.mean((noisy_latent - reencoded) ** 2))


def image_tv(img):
    img = np.asarray(img, dtype=np.float64)
    tv_h = np.mean(np.abs(img[:, 1:, :] - img[:, :-1, :]))
    tv_w = np.mean(np.abs(img[:, :, 1:] - img[:, :, :-1]))
    return float(tv_h + tv_w)


def gradient_entropy(img):
    img = np.asarray(img, dtype=np.float64)
    gh = np.abs(img[:, 1:, :] - img[:, :-1, :])
    gw = np.abs(img[:, :, 1:] - img[:, :, :-1])
    min_h = min(gh.shape[1], gw.shape[1])
    min_w = min(gh.shape[2], gw.shape[2])
    gm = np.sqrt(gh[:, :min_h, :min_w]**2 + gw[:, :min_h, :min_w]**2).ravel()
    if gm.size == 0 or np.std(gm) < 1e-12:
        return float("-inf")
    counts, _ = np.histogram(gm, bins=64, density=True)
    counts = counts[counts > 0]
    return float(-np.sum(counts * np.log(counts + 1e-12)))


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
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--alpha-high-snr", type=float, default=None)
    parser.add_argument("--alpha-switch-snr", type=float, default=-2.35)
    parser.add_argument("--watermark", choices=["component", "robust"], default="robust")
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--semantic-header-chips", type=int, default=512)
    parser.add_argument("--semantic-header-amplitude", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--out-dir", default="results/full_curves")
    args = parser.parse_args()

    image_names = [n.strip() for n in args.images.split(",")]
    snrs = np.arange(-12.0, 4.5, 0.5)
    key_snrs = [-12, -8, -4, 0, 4]  # for image grids

    rng = np.random.default_rng(args.seed)
    net, device = build_swin(args)
    ref_ldpc = PublishedThresholdLDPC()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    all_curve_data = {}

    for img_name in image_names:
        print(f"\n{'='*60}\n  {img_name}\n{'='*60}")

        img_path = Path(args.image_dir) / f"{img_name}.png"
        if not img_path.exists():
            print(f"  SKIP — not found")
            continue

        tensor, hw = load_image_tensor(img_path, device)
        original_np = tensor.cpu().numpy()[0]
        actual_shape = (hw[0] // 16, hw[1] // 16, args.c)
        candidates = sorted(set([actual_shape, (actual_shape[1], actual_shape[0], actual_shape[2])]))
        h_values = sorted({c[0] for c in candidates})
        w_values = sorted({c[1] for c in candidates})
        c_values = sorted({c[2] for c in candidates})

        with torch.inference_mode():
            net.encoder.update_resolution(*hw)
            feature = net.encoder(tensor, args.encoder_snr, args.c, args.model)
            if isinstance(feature, tuple):
                feature = feature[0]
        latent = feature.cpu().float().numpy().reshape(-1)

        code_kwargs = dict(
            length=latent.size,
            h_values=h_values,
            w_values=w_values,
            c_values=c_values,
            alpha=args.alpha,
            seed=args.seed + latent.size,
        )
        semantic_header = SemanticShapeHeader(
            h_values=h_values,
            w_values=w_values,
            c_values=c_values,
            chips_per_component=args.semantic_header_chips,
            amplitude=args.semantic_header_amplitude,
            seed=args.seed + latent.size + 777,
        )
        def make_code(alpha: float):
            local_kwargs = dict(code_kwargs)
            local_kwargs["alpha"] = alpha
            if args.watermark == "robust":
                return RobustSpreadSpectrumShapeComponents(**local_kwargs, replicas=args.replicas)
            return SpreadSpectrumShapeComponents(**local_kwargs)

        rows = []
        saved_imgs = {}  # snr -> {method: img}

        for snr_db in snrs:
            noise_seed = args.seed + int((snr_db + 12) * 100)
            noise_rng = np.random.default_rng(noise_seed)

            # --- Semantic Watermark ---
            alpha_now = args.alpha
            if args.alpha_high_snr is not None and snr_db >= args.alpha_switch_snr:
                alpha_now = args.alpha_high_snr
            code = make_code(alpha_now)
            sem_tx = code.embed_shape(latent, actual_shape)
            sem_rx = awgn(sem_tx, snr_db, noise_rng)
            sem_result = code.decode_shape(sem_rx)
            sem_ok = getattr(sem_result, "shape", None) == actual_shape
            if sem_ok:
                sem_clean = code.remove_shape(sem_rx, actual_shape)
                sem_img = decode_latent(net, sem_clean, actual_shape, snr_db, device, args.model)
                sem_psnr = psnr01(original_np, sem_img)
            else:
                sem_img = np.zeros_like(original_np)
                sem_psnr = 0.0

            # --- LDPC ---
            ldpc_ok = ref_ldpc.decode_success(snr_db)
            noise_rng2 = np.random.default_rng(noise_seed + 1)
            if ldpc_ok:
                ldpc_rx = awgn(latent, snr_db, noise_rng2)
                ldpc_img = decode_latent(net, ldpc_rx, actual_shape, snr_db, device, args.model)
                ldpc_psnr = psnr01(original_np, ldpc_img)
            else:
                ldpc_img = np.zeros_like(original_np)
                ldpc_psnr = 0.0

            # --- Semantic Shape Header (analog semantic-coded H/W/C concatenated with image latent) ---
            sh_rng = np.random.default_rng(noise_seed + 17)
            sh_tx = semantic_header.concatenate(latent, actual_shape)
            sh_rx = awgn(sh_tx, snr_db, sh_rng)
            sh_header_rx, sh_img_rx = semantic_header.split(sh_rx)
            sh_shape, _ = semantic_header.decode(sh_header_rx)
            sh_ok = sh_shape == actual_shape
            if sh_ok:
                sh_img = decode_latent(net, sh_img_rx, actual_shape, snr_db, device, args.model)
                sh_psnr = psnr01(original_np, sh_img)
            else:
                sh_img = np.zeros_like(original_np)
                sh_psnr = 0.0

            # --- No-reference methods (share one noise realization) ---
            noise_rng3 = np.random.default_rng(noise_seed + 2)
            nr_rx = awgn(latent, snr_db, noise_rng3)

            # Decode BOTH candidates once, reuse for TV, GradEnt
            dec_imgs = {}
            for s in candidates:
                try:
                    dec_imgs[s] = decode_latent(net, nr_rx, s, snr_db, device, args.model)
                except Exception:
                    dec_imgs[s] = np.zeros_like(original_np)

            # TV
            tv_scores = {s: image_tv(dec_imgs[s]) for s in candidates}
            tv_shape = min(tv_scores, key=tv_scores.get)
            tv_ok = tv_shape == actual_shape
            tv_img = dec_imgs[tv_shape]
            tv_psnr = psnr01(original_np, tv_img) if tv_ok and tv_img.shape == original_np.shape else 0.0

            # Gradient Entropy
            ge_scores = {s: gradient_entropy(dec_imgs[s]) for s in candidates}
            ge_shape = max(ge_scores, key=ge_scores.get)
            ge_ok = ge_shape == actual_shape
            ge_img = dec_imgs[ge_shape]
            ge_psnr = psnr01(original_np, ge_img) if ge_ok and ge_img.shape == original_np.shape else 0.0

            # Re-encode Check
            reenc_scores = {}
            for s in candidates:
                reenc_scores[s] = reencode_distance(net, nr_rx, s, snr_db, device, args.model, args.c)
            reenc_shape = min(reenc_scores, key=reenc_scores.get)
            reenc_ok = reenc_shape == actual_shape
            if reenc_ok and dec_imgs[reenc_shape].shape == original_np.shape:
                reenc_psnr = psnr01(original_np, dec_imgs[reenc_shape])
                reenc_img = dec_imgs[reenc_shape]
            else:
                try:
                    reenc_img = decode_latent(net, nr_rx, reenc_shape, snr_db, device, args.model)
                    reenc_psnr = psnr01(original_np, reenc_img) if reenc_img.shape == original_np.shape else 0.0
                except Exception:
                    reenc_img = np.zeros_like(original_np)
                    reenc_psnr = 0.0

            # Latent Smoothness
            ls_shape = choose_shape_by_smoothness(nr_rx, candidates)
            ls_ok = ls_shape == actual_shape
            if ls_ok and ls_shape in dec_imgs and dec_imgs[ls_shape].shape == original_np.shape:
                ls_psnr = psnr01(original_np, dec_imgs[ls_shape])
                ls_img = dec_imgs[ls_shape]
            elif ls_ok:
                try:
                    ls_img = decode_latent(net, nr_rx, ls_shape, snr_db, device, args.model)
                    ls_psnr = psnr01(original_np, ls_img) if ls_img.shape == original_np.shape else 0.0
                except Exception:
                    ls_img = np.zeros_like(original_np)
                    ls_psnr = 0.0
            else:
                ls_img = np.zeros_like(original_np)
                ls_psnr = 0.0

            rows.append({
                "image": img_name, "snr_db": snr_db,
                "semantic_psnr": sem_psnr, "semantic_ok": int(sem_ok),
                "ldpc_psnr": ldpc_psnr, "ldpc_ok": int(ldpc_ok),
                "semantic_header_psnr": sh_psnr, "semantic_header_ok": int(sh_ok),
                "reencode_psnr": reenc_psnr, "reencode_ok": int(reenc_ok),
                "latsmooth_psnr": ls_psnr, "latsmooth_ok": int(ls_ok),
                "gradent_psnr": ge_psnr, "gradent_ok": int(ge_ok),
                "tv_psnr": tv_psnr, "tv_ok": int(tv_ok),
                "watermark": args.watermark,
                "alpha": alpha_now,
            })

            # Save images for key SNRs
            if snr_db in key_snrs:
                saved_imgs[snr_db] = {
                    "original": original_np,
                    "semantic": sem_img,
                    "ldpc": ldpc_img,
                    "semantic_header": sh_img,
                    "reencode": reenc_img,
                    "latsmooth": ls_img,
                    "gradent": ge_img,
                    "tv": tv_img,
                    "sem_ok": sem_ok, "sh_ok": sh_ok, "ldpc_ok": ldpc_ok,
                    "reenc_ok": reenc_ok, "ls_ok": ls_ok,
                    "ge_ok": ge_ok, "tv_ok": tv_ok,
                }

            if snr_db % 2 == 0:
                print(f"  SNR={snr_db:+.1f}: sem={sem_psnr:.1f} semhdr={sh_psnr:.1f} ldpc={ldpc_psnr:.1f} reenc={reenc_psnr:.1f} ls={ls_psnr:.1f} ge={ge_psnr:.1f} tv={tv_psnr:.1f}")

        all_curve_data[img_name] = rows

        # --- Plot 1: PSNR vs SNR curve ---
        fig1, ax1 = plt.subplots(figsize=(12, 5.5))
        snr_vals = [r["snr_db"] for r in rows]
        methods_plot = [
            ("Semantic Watermark", "semantic_psnr", "#2196F3", "-", 2.2),
            ("Semantic Shape Header", "semantic_header_psnr", "#00BCD4", "--", 1.9),
            ("LDPC Header", "ldpc_psnr", "#F44336", "--", 2.0),
            ("Re-encode Check", "reencode_psnr", "#4CAF50", "-", 1.8),
            ("Latent Smoothness", "latsmooth_psnr", "#FF9800", "-.", 1.5),
            ("Gradient Entropy", "gradent_psnr", "#9C27B0", ":", 1.5),
            ("Image TV", "tv_psnr", "#795548", ":", 1.2),
        ]
        for label, key, color, ls, lw in methods_plot:
            vals = [r[key] for r in rows]
            ax1.plot(snr_vals, vals, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.9)

        ax1.axvline(x=-2.35, color="#F44336", linestyle=":", alpha=0.35, linewidth=1)
        ax1.text(-2.35, ax1.get_ylim()[1] * 0.97, "LDPC QEF\n−2.35 dB", fontsize=7, color="#F44336", ha="center", va="top")
        ax1.set_xlabel("Channel SNR (dB)", fontsize=11)
        ax1.set_ylabel("Utility PSNR (dB)", fontsize=11)
        ax1.set_title(f"{img_name}  ({actual_shape[0]}×{actual_shape[1]}×{actual_shape[2]})", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=7.5, loc="lower right", ncol=2)
        ax1.grid(True, alpha=0.25)
        ax1.set_xlim(-12.5, 4.5)
        fig1.tight_layout()
        fig1.savefig(out_dir / f"curve_{img_name}.png", dpi=180)
        plt.close(fig1)

        # --- Plot 2: Image grid at key SNRs ---
        n_snr = len(key_snrs)
        n_methods = 8  # Original + 7 methods
        fig2, axes2 = plt.subplots(n_snr, n_methods, figsize=(n_methods * 2.5, n_snr * 2.8))
        if n_snr == 1:
            axes2 = axes2.reshape(1, -1)

        method_labels = ["Original", "Watermark", "SemHeader", "LDPC", "ReEncode", "LatSmooth", "GradEnt", "TV"]
        method_keys = ["original", "semantic", "semantic_header", "ldpc", "reencode", "latsmooth", "gradent", "tv"]
        method_ok_keys = [None, "sem_ok", "sh_ok", "ldpc_ok", "reenc_ok", "ls_ok", "ge_ok", "tv_ok"]

        for row_idx, snr_db in enumerate(key_snrs):
            data = saved_imgs.get(snr_db)
            if data is None:
                continue
            for col_idx, (mlabel, mkey, mok) in enumerate(zip(method_labels, method_keys, method_ok_keys)):
                ax = axes2[row_idx, col_idx]
                img = data[mkey]
                disp = np.clip(img.transpose(1, 2, 0), 0, 1)
                ax.imshow(disp)
                ok_str = ""
                if mok and mkey != "original":
                    ok_str = " ✓" if data[mok] else " ✗"
                ax.set_title(f"{mlabel}{ok_str}", fontsize=7)
                ax.axis("off")

        for row_idx, snr_db in enumerate(key_snrs):
            axes2[row_idx, 0].set_ylabel(f"SNR {snr_db:+.0f} dB", fontsize=9, fontweight="bold", rotation=90, labelpad=12)

        fig2.suptitle(f"{img_name} — Reconstruction Comparison", fontsize=13, fontweight="bold", y=1.01)
        fig2.tight_layout()
        fig2.savefig(out_dir / f"grid_{img_name}.png", dpi=180)
        plt.close(fig2)
        print(f"  saved curve + grid for {img_name}")

    # --- Save all CSV ---
    all_rows = []
    for rows in all_curve_data.values():
        all_rows.extend(rows)
    csv_path = out_dir / "full_psnr_curves.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Aggregate curve (average across images) ---
    print("\n=== Aggregate ===")
    fig_agg, ax_agg = plt.subplots(figsize=(12, 5.5))
    snr_set = sorted(set(r["snr_db"] for rows in all_curve_data.values() for r in rows))
    agg = {
        k: []
        for k in [
            "semantic_psnr",
            "semantic_header_psnr",
            "ldpc_psnr",
            "reencode_psnr",
            "latsmooth_psnr",
            "gradent_psnr",
            "tv_psnr",
        ]
    }
    for snr_db in snr_set:
        sub = [r for rows in all_curve_data.values() for r in rows if r["snr_db"] == snr_db]
        for k in agg:
            vals = [r[k] for r in sub]
            agg[k].append(float(np.mean(vals)))

    methods_agg = [
        ("Semantic Watermark", "semantic_psnr", "#2196F3", "-", 2.2),
        ("Semantic Shape Header", "semantic_header_psnr", "#00BCD4", "--", 1.9),
        ("LDPC Header", "ldpc_psnr", "#F44336", "--", 2.0),
        ("Re-encode Check", "reencode_psnr", "#4CAF50", "-", 1.8),
        ("Latent Smoothness", "latsmooth_psnr", "#FF9800", "-.", 1.5),
        ("Gradient Entropy", "gradent_psnr", "#9C27B0", ":", 1.5),
        ("Image TV", "tv_psnr", "#795548", ":", 1.2),
    ]
    for label, key, color, ls, lw in methods_agg:
        ax_agg.plot(snr_set, agg[key], color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.9)
    ax_agg.axvline(x=-2.35, color="#F44336", linestyle=":", alpha=0.35, linewidth=1)
    ax_agg.text(-2.35, ax_agg.get_ylim()[1] * 0.97, "LDPC QEF\n−2.35 dB", fontsize=7, color="#F44336", ha="center", va="top")
    ax_agg.set_xlabel("Channel SNR (dB)"); ax_agg.set_ylabel("Utility PSNR (dB)")
    ax_agg.set_title(f"Average Utility PSNR Across {len(all_curve_data)} Kodak Images", fontsize=13, fontweight="bold")
    ax_agg.legend(fontsize=8, loc="lower right", ncol=2); ax_agg.grid(True, alpha=0.25)
    ax_agg.set_xlim(-12.5, 4.5)
    fig_agg.tight_layout()
    fig_agg.savefig(out_dir / "curve_aggregate.png", dpi=200)
    plt.close(fig_agg)

    print(f"\nAll outputs in {out_dir}/")
    print(f"  curve_*.png  — per-image PSNR curves")
    print(f"  grid_*.png   — per-image reconstruction grids")
    print(f"  curve_aggregate.png — average across images")
    print(f"  full_psnr_curves.csv — raw data")


if __name__ == "__main__":
    main()
