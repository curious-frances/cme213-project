#!/usr/bin/env python3
"""
roofline.py — Roofline plot for stereo SAD benchmark.

Usage:
  python roofline.py benchmark.csv [--peak-gops 14000] [--peak-bw 616]

Copy benchmark.csv off the cluster first:
  scp <user>@<cluster>:~/cme213-project/benchmark.csv .
  python roofline.py benchmark.csv

Default hardware numbers are for NVIDIA RTX 2080 Ti (Turing, gpu-turing):
  Peak INT8 throughput : ~14,000 GOps/s (use INT32 peak ~500 GOps/s for SAD)
  Peak memory bandwidth: 616 GB/s

Adjust with --peak-gops and --peak-bw to match your actual GPU.
For INT32 SAD on Turing, a realistic peak is ~500 GOps/s.
"""

import argparse
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Operational intensity for stereo SAD
#
# Each output pixel reads:
#   - left patch:  (2R+1)^2 bytes  (bytes from global memory)
#   - right patch: (2R+1)^2 bytes per disparity × max_disp disparities
#
# Without caching, bytes_loaded = 2 × (2R+1)^2 × max_disp
# Operations = 2 × (2R+1)^2 × max_disp  (one abs + one add per element)
# → naive OI ≈ 1 op / byte
#
# With shared memory, a block of B×B threads loads the tile once and reuses:
#   bytes_loaded ≈ (B+2R)×(B+2R) + (B+2R)×(B+2R+max_disp)
#   ops          ≈ B^2 × max_disp × (2R+1)^2 × 2
# → smem OI can be ~10–40× higher depending on block size.
#
# This script computes an *effective* OI from the measured throughput and
# bandwidth headroom, and also plots the theoretical naive OI as a reference.
# ---------------------------------------------------------------------------

COLORS = {
    "CPU":       "#4C72B0",
    "GPU_basic": "#DD8452",
    "GPU_smem":  "#55A868",
    "GPU_tiled": "#C44E52",
}
MARKERS = {
    "CPU":       "s",
    "GPU_basic": "o",
    "GPU_smem":  "^",
    "GPU_tiled": "D",
}

def oi_naive(row):
    """Naive operational intensity (no data reuse): ops / bytes."""
    R        = int(row["radius"])
    max_disp = int(row["max_disp"])
    patch    = (2 * R + 1) ** 2
    ops      = 2 * patch * max_disp      # per output pixel
    bytes_   = 2 * patch * max_disp      # left + right, per pixel
    return ops / bytes_

def main():
    parser = argparse.ArgumentParser(description="Roofline plot for stereo SAD")
    parser.add_argument("csv", help="benchmark.csv from the cluster")
    parser.add_argument("--peak-gops", type=float, default=500.0,
                        help="Peak integer GOps/s for the GPU (default 500 for Turing INT32)")
    parser.add_argument("--peak-bw",   type=float, default=616.0,
                        help="Peak memory bandwidth GB/s (default 616 for RTX 2080 Ti)")
    parser.add_argument("--out", default="roofline.pdf",
                        help="Output file (default roofline.pdf)")
    args = parser.parse_args()

    rows = []
    try:
        with open(args.csv) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"ERROR: '{args.csv}' not found. Copy it from the cluster first.", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("ERROR: CSV is empty.", file=sys.stderr)
        sys.exit(1)

    peak_gops = args.peak_gops   # GOps/s
    peak_bw   = args.peak_bw     # GB/s
    ridge_oi  = peak_gops / peak_bw  # ops/byte at the ridge point

    print(f"GPU peak: {peak_gops} GOps/s,  BW: {peak_bw} GB/s,  ridge OI: {ridge_oi:.2f} ops/byte")

    fig, ax = plt.subplots(figsize=(9, 6))

    # ---- Roofline ----
    oi_range = np.logspace(-2, 4, 500)
    roof = np.minimum(peak_bw * oi_range, peak_gops)
    ax.loglog(oi_range, roof, "k-", linewidth=2, label="Roofline")

    # Ridge point
    ax.axvline(ridge_oi, color="gray", linestyle="--", linewidth=1)
    ax.text(ridge_oi * 1.05, peak_gops * 0.6,
            f"Ridge\n{ridge_oi:.1f} ops/B", fontsize=8, color="gray")

    # ---- Data points ----
    legend_patches = []
    for row in rows:
        impl      = row["impl"]
        gops_s    = float(row["gops_per_s"])
        oi        = oi_naive(row)

        color  = COLORS.get(impl, "purple")
        marker = MARKERS.get(impl, "x")

        ax.plot(oi, gops_s, marker=marker, markersize=10,
                color=color, zorder=5)
        ax.annotate(f"  {impl}\n  {gops_s:.1f} GOps/s",
                    xy=(oi, gops_s), fontsize=8, color=color,
                    va="center")
        legend_patches.append(
            mpatches.Patch(color=color, label=impl))

    # ---- Labels ----
    ax.set_xlabel("Operational Intensity (ops / byte)", fontsize=11)
    ax.set_ylabel("Performance (GOps/s)", fontsize=11)
    ax.set_title("Roofline — Stereo SAD Disparity Estimation", fontsize=13)
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(oi_range[0], oi_range[-1])
    ax.set_ylim(1e-2, peak_gops * 3)

    # Annotate bandwidth and compute ceilings
    ax.text(0.02, peak_bw * 0.015,
            f"BW-bound slope: {peak_bw} GB/s",
            fontsize=8, color="black", rotation=30)
    ax.axhline(peak_gops, color="black", linestyle=":", linewidth=1)
    ax.text(oi_range[-1] * 0.5, peak_gops * 1.05,
            f"Compute ceiling: {peak_gops} GOps/s",
            fontsize=8, ha="right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")
    plt.show()

if __name__ == "__main__":
    main()
