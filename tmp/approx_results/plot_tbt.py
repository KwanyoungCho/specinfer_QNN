#!/usr/bin/env python3
"""Plot per-step TBT (time-between-tokens) timeseries for baseline vs approximate sampling.

Reads CSVs produced by speculative-eagle-2(-approx) --tbt-csv.
Each row: step_idx,n_tokens,step_total_us,tbt_per_token_us,cum_tokens,cum_us
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
RUNS = [
    ("baseline.csv", "baseline (exact greedy, eagle-2)", "#1f77b4"),
    ("approx_k1.csv", "approx k=1 (= exact)", "#9467bd"),
    ("approx_k2.csv", "approx k=2", "#2ca02c"),
    ("approx_k3.csv", "approx k=3", "#ff7f0e"),
    ("approx_k5.csv", "approx k=5", "#d62728"),
]


def load(path):
    df = pd.read_csv(path)
    df["tbt_per_token_ms"] = df["tbt_per_token_us"] / 1000.0
    df["cum_ms"] = df["cum_us"] / 1000.0
    return df


def main():
    runs = []
    for fn, label, color in RUNS:
        p = HERE / fn
        if not p.exists():
            print(f"skip missing: {p}")
            continue
        df = load(p)
        runs.append((label, color, df))

    if not runs:
        print("no input csv found")
        return

    # ---------------- Plot 1: TBT vs generated-token index (wall-clock TBT curve) ----------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, color, df in runs:
        # x = cumulative generated tokens (so each step appears at its end-of-step cum_tokens)
        # y = per-step TBT (ms/token) — drop step 0 (includes prefill warm-up)
        if len(df) > 1:
            sub = df.iloc[1:]
            ax.plot(sub["cum_tokens"], sub["tbt_per_token_ms"], marker="o", ms=3, lw=1.2,
                    label=label, color=color, alpha=0.85)
    ax.set_xlabel("Generated token index")
    ax.set_ylabel("Time-between-tokens (ms/token, per-step)")
    ax.set_title("Per-step TBT over time (decode phase)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "tbt_timeseries.png", dpi=150)
    print(f"wrote {HERE/'tbt_timeseries.png'}")

    # ---------------- Plot 2: cumulative wall-clock time vs tokens generated ----------------
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, color, df in runs:
        ax.plot(df["cum_tokens"], df["cum_ms"], lw=1.5, label=label, color=color)
    ax.set_xlabel("Generated token index")
    ax.set_ylabel("Cumulative decode time (ms)")
    ax.set_title("Cumulative wall-clock decode time (slope = sec/token)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "cumulative_decode.png", dpi=150)
    print(f"wrote {HERE/'cumulative_decode.png'}")

    # ---------------- Plot 3: TBT distribution (histogram / box) ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    data = []
    labels = []
    colors = []
    for label, color, df in runs:
        if len(df) <= 1:
            continue
        sub = df.iloc[1:]
        data.append(sub["tbt_per_token_ms"].values)
        labels.append(label)
        colors.append(color)
    # histogram (overlay)
    for d, lbl, c in zip(data, labels, colors):
        ax1.hist(d, bins=25, alpha=0.45, label=lbl, color=c)
    ax1.set_xlabel("TBT (ms/token)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("TBT distribution (per-step, excl. step 0)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    # boxplot
    bp = ax2.boxplot(data, labels=[l.split()[0] + ("\n" + l.split()[1] if len(l.split()) > 1 else "") for l in labels],
                     patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
    ax2.set_ylabel("TBT (ms/token)")
    ax2.set_title("TBT spread per run")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "tbt_distribution.png", dpi=150)
    print(f"wrote {HERE/'tbt_distribution.png'}")

    # ---------------- Summary table ----------------
    print("\n==== summary ====")
    print(f"{'run':<30} {'TPS':>8} {'avg_len':>8} {'mean_tbt':>10} {'std_tbt':>9} {'max_tbt':>9}")
    for label, _color, df in runs:
        if len(df) <= 1:
            continue
        sub = df.iloc[1:]
        total_us = df["cum_us"].iloc[-1]
        total_tokens = df["cum_tokens"].iloc[-1]
        tps = total_tokens / (total_us / 1e6)
        avg_len = df["n_tokens"].iloc[1:].mean()
        mean_tbt = sub["tbt_per_token_ms"].mean()
        std_tbt = sub["tbt_per_token_ms"].std()
        max_tbt = sub["tbt_per_token_ms"].max()
        print(f"{label:<30} {tps:8.3f} {avg_len:8.3f} {mean_tbt:10.3f} {std_tbt:9.3f} {max_tbt:9.3f}")


if __name__ == "__main__":
    main()
