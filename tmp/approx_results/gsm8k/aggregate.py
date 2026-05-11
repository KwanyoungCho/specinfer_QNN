#!/usr/bin/env python3
"""Aggregate GSM8K sweep results.

Reads per-run logs and TBT CSVs in this directory, builds summary tables
and plots comparing baseline vs approx (k=1,2,3,5) over 20 prompts.
"""

import re
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CONFIGS = ["baseline", "k1", "k2", "k3", "k5"]
COLORS = {
    "baseline": "#1f77b4",
    "k1": "#9467bd",
    "k2": "#2ca02c",
    "k3": "#ff7f0e",
    "k5": "#d62728",
}


def parse_log(path: Path):
    """Pull summary metrics out of a single run's stdout log."""
    text = path.read_text(errors="ignore")
    out = {}
    m = re.search(r"decoded\s+(\d+)\s+tokens in\s+([\d.]+)\s+seconds,\s+speed:\s+([\d.]+)", text)
    if m:
        out["n_predict"] = int(m.group(1))
        out["decode_s"] = float(m.group(2))
        out["tps"] = float(m.group(3))
    m = re.search(r"n_drafted = (\d+)", text)
    if m:
        out["n_drafted"] = int(m.group(1))
    m = re.search(r"n_accept\s*=\s*(\d+)", text)
    if m:
        out["n_accept"] = int(m.group(1))
    m = re.search(r"accept\s*=\s*([\d.]+)%", text)
    if m:
        out["accept_pct"] = float(m.group(1))
    m = re.search(r"Avg length:\s+([\d.]+)", text)
    if m:
        out["avg_len"] = float(m.group(1))
    m = re.search(r"mean TBT\s*:\s*([\d.]+)", text)
    if m:
        out["mean_tbt_ms"] = float(m.group(1))
    m = re.search(r"stddev TBT\s*:\s*([\d.]+)", text)
    if m:
        out["std_tbt_ms"] = float(m.group(1))
    m = re.search(r"min  TBT\s*:\s*([\d.]+)", text)
    if m:
        out["min_tbt_ms"] = float(m.group(1))
    m = re.search(r"max  TBT\s*:\s*([\d.]+)", text)
    if m:
        out["max_tbt_ms"] = float(m.group(1))
    return out


def main():
    rows = []
    for p in range(1, 21):
        pid = f"{p:02d}"
        for cfg in CONFIGS:
            log = HERE / f"{cfg}_p{pid}.log"
            if not log.exists():
                continue
            metrics = parse_log(log)
            metrics["config"] = cfg
            metrics["prompt"] = pid
            rows.append(metrics)

    if not rows:
        print("no logs found yet")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["config", "prompt"]).reset_index(drop=True)
    df.to_csv(HERE / "summary_per_run.csv", index=False)
    print(f"wrote {HERE/'summary_per_run.csv'} ({len(df)} runs)")

    # Per-config aggregate
    agg_cols = ["n_predict", "tps", "n_accept", "accept_pct", "avg_len",
                "mean_tbt_ms", "std_tbt_ms", "min_tbt_ms", "max_tbt_ms"]
    agg = df.groupby("config")[agg_cols].agg(["mean", "std"]).round(3)
    agg.to_csv(HERE / "summary_by_config.csv")
    print(f"wrote {HERE/'summary_by_config.csv'}")

    print("\n=== per-config means ===")
    print(df.groupby("config")[agg_cols].mean().round(3).to_string())

    # ---------- Plot 1: TPS per prompt (grouped bars) ----------
    fig, ax = plt.subplots(figsize=(14, 5))
    prompts = sorted(df["prompt"].unique())
    x = np.arange(len(prompts))
    width = 0.16
    for i, cfg in enumerate(CONFIGS):
        sub = df[df["config"] == cfg].set_index("prompt").reindex(prompts)
        ax.bar(x + (i - 2) * width, sub["tps"], width, label=cfg, color=COLORS[cfg])
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.set_xlabel("Prompt ID")
    ax.set_ylabel("Decode TPS (tokens/sec)")
    ax.set_title("Decode TPS per prompt — GSM8K (vicuna + EAGLE)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "tps_per_prompt.png", dpi=150)
    print(f"wrote {HERE/'tps_per_prompt.png'}")

    # ---------- Plot 2: accept length per prompt ----------
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cfg in enumerate(CONFIGS):
        sub = df[df["config"] == cfg].set_index("prompt").reindex(prompts)
        ax.bar(x + (i - 2) * width, sub["avg_len"], width, label=cfg, color=COLORS[cfg])
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.set_xlabel("Prompt ID")
    ax.set_ylabel("Avg accept length (tokens/step)")
    ax.set_title("Average accepted length per prompt")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "accept_len_per_prompt.png", dpi=150)
    print(f"wrote {HERE/'accept_len_per_prompt.png'}")

    # ---------- Plot 3: TBT mean ± std bar plot ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    summary = df.groupby("config")[agg_cols].mean().reindex(CONFIGS)
    summary_std = df.groupby("config")[agg_cols].std().reindex(CONFIGS)
    ax1.bar(summary.index, summary["mean_tbt_ms"], yerr=summary_std["mean_tbt_ms"],
            color=[COLORS[c] for c in summary.index], alpha=0.7, capsize=4)
    ax1.set_ylabel("Mean TBT (ms/token) — avg across prompts")
    ax1.set_title("Average TBT per config (lower = faster)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.bar(summary.index, summary["std_tbt_ms"], yerr=summary_std["std_tbt_ms"],
            color=[COLORS[c] for c in summary.index], alpha=0.7, capsize=4)
    ax2.set_ylabel("Stddev of TBT (ms/token) — avg across prompts")
    ax2.set_title("TBT variance per config (lower = smoother)")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "tbt_aggregate.png", dpi=150)
    print(f"wrote {HERE/'tbt_aggregate.png'}")

    # ---------- Plot 4: TBT timeseries (one example prompt + cross-prompt mean) ----------
    # find a prompt that completed in all configs
    pivot = df.pivot(index="prompt", columns="config", values="n_predict")
    valid = pivot.dropna()
    if not valid.empty:
        example_p = valid.index[len(valid) // 2]  # middle prompt
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for cfg in CONFIGS:
            csv_path = HERE / "csv_raw" / f"{cfg}_p{example_p}.csv"
            if not csv_path.exists():
                csv_path = HERE / f"{cfg}_p{example_p}.csv"
            if csv_path.exists():
                d = pd.read_csv(csv_path)
                if len(d) > 1:
                    d2 = d.iloc[1:]
                    ax.plot(d2["cum_tokens"], d2["tbt_per_token_us"] / 1000.0,
                            marker="o", ms=2, lw=1, label=cfg, color=COLORS[cfg], alpha=0.8)
        ax.set_xlabel("Generated token index")
        ax.set_ylabel("TBT (ms/token, per-step)")
        ax.set_title(f"Per-step TBT timeseries — prompt {example_p}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(HERE / f"tbt_timeseries_p{example_p}.png", dpi=150)
        print(f"wrote {HERE/f'tbt_timeseries_p{example_p}.png'}")

    # ---------- Plot 5: scatter — n_predict vs TPS ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in CONFIGS:
        sub = df[df["config"] == cfg]
        ax.scatter(sub["n_predict"], sub["tps"], label=cfg, color=COLORS[cfg], s=40, alpha=0.7)
    ax.set_xlabel("Tokens generated (n_predict)")
    ax.set_ylabel("Decode TPS")
    ax.set_title("TPS vs generation length (each dot = one prompt)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "tps_vs_length.png", dpi=150)
    print(f"wrote {HERE/'tps_vs_length.png'}")


if __name__ == "__main__":
    main()
