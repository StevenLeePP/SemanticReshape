"""Resource-matched shape metadata comparison.

This script answers the reviewer concern that a semantic watermark spends
latent distortion, while an ultra-low-rate conventional metadata code could
spend the same order of resource as explicit side symbols.

It reuses the already computed SwinJSCC oracle/watermark curves and evaluates
side-header metadata reliability under equal side-symbol budgets:

- analog semantic H/W/C spread header;
- conventional 3-bit BPSK repetition code;
- optimistic random BPSK codebook with ML correlation decoding.

For side headers, practical utility PSNR is:

    P(shape recovered) * PSNR_oracle_known_shape

because the image latent itself is not modified; a failed shape decode means
the receiver cannot reshape the frame and utility is counted as zero.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import erfc, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sdlatent.redundant_header import (
    RandomCodebookShapeHeader,
    RepetitionCodedShapeHeader,
    add_awgn_with_reference_power,
)
from sdlatent.semantic_header import SemanticShapeHeader


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def qfunc(x: float) -> float:
    return 0.5 * erfc(x / sqrt(2.0))


def repetition_success_probability(header: RepetitionCodedShapeHeader, snr_db: float) -> float:
    snr_linear = 10.0 ** (snr_db / 10.0)
    success = 1.0
    for count in header.counts:
        bit_error = qfunc(sqrt(snr_linear * int(count)))
        success *= 1.0 - bit_error
    return float(success)


def discover_shapes(image_dir: Path, image_names: list[str], c: int) -> dict[str, tuple[int, int, int]]:
    from PIL import Image

    shapes = {}
    for name in image_names:
        path = image_dir / f"{name}.png"
        width, height = Image.open(path).size
        crop_w = width - width % 128
        crop_h = height - height % 128
        shapes[name] = (crop_h // 16, crop_w // 16, c)
    return shapes


def aggregate_base_curves(rows: list[dict]) -> dict[str, dict[float, float]]:
    keys = ["semantic_psnr", "ldpc_psnr"]
    out = {k: {} for k in keys}
    snrs = sorted({float(r["snr_db"]) for r in rows})
    for snr_db in snrs:
        sub = [r for r in rows if float(r["snr_db"]) == snr_db]
        for key in keys:
            out[key][snr_db] = float(np.mean([float(r[key]) for r in sub]))
    return out


def aggregate_oracle(rows: list[dict]) -> dict[float, float]:
    out = {}
    snrs = sorted({float(r["snr_db"]) for r in rows})
    for snr_db in snrs:
        vals = [float(r["oracle_psnr"]) for r in rows if float(r["snr_db"]) == snr_db]
        out[snr_db] = float(np.mean(vals))
    return out


def oracle_by_image(rows: list[dict]) -> dict[tuple[str, float], float]:
    return {(r["image"], float(r["snr_db"])): float(r["oracle_psnr"]) for r in rows}


def estimate_semantic_header_success(
    header: SemanticShapeHeader,
    shape: tuple[int, int, int],
    snr_db: float,
    trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    tx = header.encode(shape, host_power=1.0)
    ok = 0
    for _ in range(trials):
        rx = add_awgn_with_reference_power(tx, snr_db, 1.0, rng)
        decoded, _ = header.decode(rx)
        ok += int(decoded == shape)
    return ok / trials


def estimate_codebook_success(
    header: RandomCodebookShapeHeader,
    shape_id: int,
    snr_db: float,
    trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    tx = header.encode(shape_id, host_power=1.0)
    ok = 0
    for _ in range(trials):
        rx = add_awgn_with_reference_power(tx, snr_db, 1.0, rng)
        decoded = header.decode(rx)
        ok += int(decoded.shape_id == shape_id)
    return ok / trials


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_overlay(rows: list[dict], out_dir: Path, focus_symbols: int) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6))
    styles = {
        "oracle": ("Oracle known shape", "black", "-", 2.5),
        "semantic_watermark": ("Semantic watermark, 0 side symbols", "#2196F3", "-", 2.2),
        "dvbs2_ldpc": ("DVB-S2 QPSK 1/4 threshold header", "#F44336", "--", 2.0),
        "semantic_header": (f"Analog semantic H/W/C header, N={focus_symbols}", "#00BCD4", "--", 2.2),
        "repetition_code": (f"Traditional 3-bit repetition code, N={focus_symbols}", "#E91E63", "-.", 2.2),
        "random_codebook": (f"Traditional ML random codebook, N={focus_symbols}", "#673AB7", ":", 2.0),
    }
    for method, (label, color, ls, lw) in styles.items():
        sub = [
            r for r in rows
            if r["method"] == method and int(float(r["symbols"])) in (0, focus_symbols, 192)
        ]
        if method in {"semantic_header", "repetition_code", "random_codebook"}:
            sub = [r for r in sub if int(float(r["symbols"])) == focus_symbols]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: float(r["snr_db"]))
        ax.plot(
            [float(r["snr_db"]) for r in sub],
            [float(r["utility_psnr"]) for r in sub],
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=label,
            alpha=0.9,
        )
    ax.axvline(-2.35, color="#F44336", linestyle=":", alpha=0.35)
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("Utility PSNR (dB)")
    ax.set_title("Resource-Matched Shape Metadata: PSNR vs SNR")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "resource_matched_psnr_overlay.png", dpi=220)
    plt.close(fig)


def plot_budget_panels(rows: list[dict], budgets: list[int], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    colors = {
        "semantic_header": "#00BCD4",
        "repetition_code": "#E91E63",
        "random_codebook": "#673AB7",
        "semantic_watermark": "#2196F3",
        "dvbs2_ldpc": "#F44336",
        "oracle": "black",
    }
    labels = {
        "semantic_header": "Analog semantic H/W/C",
        "repetition_code": "Traditional repetition",
        "random_codebook": "Traditional ML codebook",
        "semantic_watermark": "Watermark",
        "dvbs2_ldpc": "DVB-S2 LDPC",
        "oracle": "Oracle",
    }
    linestyles = {
        "semantic_header": "--",
        "repetition_code": "-.",
        "random_codebook": ":",
        "semantic_watermark": "-",
        "dvbs2_ldpc": "--",
        "oracle": "-",
    }
    for ax, budget in zip(axes, budgets):
        for method in ["oracle", "semantic_watermark", "dvbs2_ldpc", "semantic_header", "repetition_code", "random_codebook"]:
            if method in {"semantic_header", "repetition_code", "random_codebook"}:
                sub = [r for r in rows if r["method"] == method and int(float(r["symbols"])) == budget]
            else:
                sub = [r for r in rows if r["method"] == method]
            sub = sorted(sub, key=lambda r: float(r["snr_db"]))
            ax.plot(
                [float(r["snr_db"]) for r in sub],
                [float(r["utility_psnr"]) for r in sub],
                color=colors[method],
                linestyle=linestyles[method],
                linewidth=2.0 if method != "oracle" else 2.4,
                alpha=0.9,
                label=labels[method],
            )
        ax.set_title(f"Side metadata budget N={budget} symbols")
        ax.grid(True, alpha=0.22)
        ax.axvline(-2.35, color="#F44336", linestyle=":", alpha=0.25)
    axes[0].legend(fontsize=7.5, loc="lower right")
    for ax in axes[2:]:
        ax.set_xlabel("SNR (dB)")
    for ax in axes[::2]:
        ax.set_ylabel("Utility PSNR (dB)")
    fig.suptitle("Equal-Bandwidth Metadata Baselines vs Zero-Side-Bandwidth Watermark", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "resource_matched_psnr_by_budget.png", dpi=220)
    plt.close(fig)


def plot_success(rows: list[dict], budgets: list[int], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    methods = [
        ("semantic_header", "Analog semantic H/W/C", "#00BCD4"),
        ("repetition_code", "Traditional repetition", "#E91E63"),
        ("random_codebook", "Traditional ML codebook", "#673AB7"),
    ]
    for ax, (method, label, color) in zip(axes, methods):
        for budget in budgets:
            sub = sorted(
                [r for r in rows if r["method"] == method and int(float(r["symbols"])) == budget],
                key=lambda r: float(r["snr_db"]),
            )
            ax.plot(
                [float(r["snr_db"]) for r in sub],
                [float(r["shape_success"]) for r in sub],
                linewidth=1.8,
                label=f"N={budget}",
            )
        ax.axvline(-2.35, color="#F44336", linestyle=":", alpha=0.3)
        ax.set_title(label)
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Shape recovery success")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "resource_matched_shape_success.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="../SwinJSCC/Kodak_dataset")
    parser.add_argument("--c", type=int, default=96)
    parser.add_argument("--base-curves", default="results/final_resource/base_full_psnr_curves.csv")
    parser.add_argument("--oracle-csv", default="results/final_resource/oracle_upper_bound.csv")
    parser.add_argument("--out-dir", default="results/final_resource")
    parser.add_argument("--budgets", default="96,300,768,1536")
    parser.add_argument("--metadata-bits", type=int, default=3)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--focus-symbols", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_csv(Path(args.base_curves))
    oracle_rows = load_csv(Path(args.oracle_csv))
    base = aggregate_base_curves(base_rows)
    oracle_mean = aggregate_oracle(oracle_rows)
    oracle_img = oracle_by_image(oracle_rows)
    snrs = sorted(set(oracle_mean) & set(base["semantic_psnr"]))
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]

    image_names = sorted({r["image"] for r in oracle_rows})
    image_shapes = discover_shapes(Path(args.image_dir), image_names, args.c)
    shape_book = sorted(set(image_shapes.values()))
    shape_to_id = {shape: i for i, shape in enumerate(shape_book)}
    h_values = sorted({s[0] for s in shape_book})
    w_values = sorted({s[1] for s in shape_book})
    c_values = sorted({s[2] for s in shape_book})

    rows: list[dict] = []
    for snr_db in snrs:
        rows.append({
            "method": "oracle",
            "symbols": 0,
            "snr_db": snr_db,
            "shape_success": 1.0,
            "utility_psnr": oracle_mean[snr_db],
            "note": "known shape upper bound",
        })
        rows.append({
            "method": "semantic_watermark",
            "symbols": 0,
            "snr_db": snr_db,
            "shape_success": np.mean([float(r["semantic_ok"]) for r in base_rows if float(r["snr_db"]) == snr_db]),
            "utility_psnr": base["semantic_psnr"][snr_db],
            "note": "zero side symbols; alpha=0.10 latent watermark",
        })
        rows.append({
            "method": "dvbs2_ldpc",
            "symbols": 192,
            "snr_db": snr_db,
            "shape_success": np.mean([float(r["ldpc_ok"]) for r in base_rows if float(r["snr_db"]) == snr_db]),
            "utility_psnr": base["ldpc_psnr"][snr_db],
            "note": "published DVB-S2 QPSK 1/4 threshold model",
        })

    success_cache: dict[tuple[str, int, tuple[int, int, int] | int, float], float] = {}
    for budget in budgets:
        semantic_header = SemanticShapeHeader(
            h_values=h_values,
            w_values=w_values,
            c_values=c_values,
            chips_per_component=budget // 3,
            amplitude=1.0,
            seed=args.seed + budget,
        )
        repetition_header = RepetitionCodedShapeHeader(
            num_classes=len(shape_book),
            total_symbols=budget,
            metadata_bits=args.metadata_bits,
            amplitude=1.0,
        )
        codebook_header = RandomCodebookShapeHeader(
            num_classes=len(shape_book),
            total_symbols=budget,
            metadata_bits=args.metadata_bits,
            amplitude=1.0,
            seed=args.seed + 17 * budget,
        )

        for snr_db in snrs:
            method_success_by_image = defaultdict(dict)
            for shape in shape_book:
                seed_base = args.seed + int((snr_db + 100.0) * 100) + budget * 31 + shape_to_id[shape]
                success_cache[("semantic_header", budget, shape, snr_db)] = estimate_semantic_header_success(
                    semantic_header, shape, snr_db, args.trials, seed_base
                )
                success_cache[("repetition_code", budget, shape_to_id[shape], snr_db)] = repetition_success_probability(
                    repetition_header, snr_db
                )
                success_cache[("random_codebook", budget, shape_to_id[shape], snr_db)] = estimate_codebook_success(
                    codebook_header, shape_to_id[shape], snr_db, args.trials, seed_base + 999
                )

            for method in ["semantic_header", "repetition_code", "random_codebook"]:
                psnr_vals = []
                ok_vals = []
                for image in image_names:
                    shape = image_shapes[image]
                    shape_id = shape_to_id[shape]
                    key_shape: tuple[int, int, int] | int = shape if method == "semantic_header" else shape_id
                    success = success_cache[(method, budget, key_shape, snr_db)]
                    ok_vals.append(success)
                    psnr_vals.append(success * oracle_img[(image, snr_db)])
                rows.append({
                    "method": method,
                    "symbols": budget,
                    "snr_db": snr_db,
                    "shape_success": float(np.mean(ok_vals)),
                    "utility_psnr": float(np.mean(psnr_vals)),
                    "note": f"{args.metadata_bits}-bit shape metadata; rate={args.metadata_bits / budget:.5f}",
                })

    write_rows(out_dir / "resource_matched_psnr.csv", rows)
    plot_overlay(rows, out_dir, args.focus_symbols)
    plot_budget_panels(rows, budgets, out_dir)
    plot_success(rows, budgets, out_dir)

    summary = out_dir / "resource_matched_summary.md"
    with summary.open("w") as f:
        f.write("# Resource-Matched Metadata Baselines\n\n")
        f.write("This experiment compares zero-side-bandwidth semantic watermarking against side-symbol metadata codes under equal side-symbol budgets.\n\n")
        f.write(f"- shape book: `{shape_book}`\n")
        f.write(f"- metadata bits for traditional code: `{args.metadata_bits}`\n")
        f.write(f"- budgets: `{budgets}` real channel symbols\n")
        f.write(f"- trials for analog/ML header success: `{args.trials}` per shape/SNR/budget\n\n")
        f.write("Important interpretation: the semantic watermark spends no extra symbols but injects latent distortion. The side-header baselines spend explicit bandwidth and do not distort the image latent, so their utility equals oracle PSNR when the metadata decodes correctly.\n\n")
        f.write("## Focus: 300-symbol rate-1/100-style header\n\n")
        f.write("| SNR (dB) | Watermark PSNR | DVB-S2 LDPC PSNR | Analog Semantic Header PSNR | Traditional Repetition PSNR | Traditional ML Codebook PSNR |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for snr_db in [-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0]:
            vals = {}
            for method in ["semantic_watermark", "dvbs2_ldpc", "semantic_header", "repetition_code", "random_codebook"]:
                sub = [
                    r for r in rows
                    if r["method"] == method
                    and float(r["snr_db"]) == snr_db
                    and (method not in {"semantic_header", "repetition_code", "random_codebook"} or int(r["symbols"]) == args.focus_symbols)
                ]
                vals[method] = float(sub[0]["utility_psnr"]) if sub else float("nan")
            f.write(
                f"| {snr_db:.0f} | {vals['semantic_watermark']:.2f} | {vals['dvbs2_ldpc']:.2f} | "
                f"{vals['semantic_header']:.2f} | {vals['repetition_code']:.2f} | {vals['random_codebook']:.2f} |\n"
            )
        f.write("\n## Conclusion\n\n")
        f.write("The reviewer concern is valid in a bandwidth-allowed setting: a very low-rate conventional side code can remove the LDPC cliff and closely track the oracle at the tested SNRs. The semantic watermark should therefore be positioned as a zero-extra-side-symbol self-describing latent mechanism, not as universally superior to arbitrarily redundant metadata channels.\n")

    print(f"Saved {out_dir / 'resource_matched_psnr.csv'}")
    print(f"Saved {out_dir / 'resource_matched_psnr_overlay.png'}")
    print(f"Saved {out_dir / 'resource_matched_psnr_by_budget.png'}")
    print(f"Saved {out_dir / 'resource_matched_shape_success.png'}")
    print(f"Saved {summary}")


if __name__ == "__main__":
    main()
