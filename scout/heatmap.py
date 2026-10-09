"""Finviz-style sector treemap: tile size = market cap, colour = 1-month return."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import squarify  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("rg", ["#c0392b", "#3a3f4b", "#27ae60"])
CLIP = 0.15  # colour saturates at ±15%


def sector_heatmap(scores: pd.DataFrame, fund: pd.DataFrame, path: str, asof: pd.Timestamp) -> str | None:
    df = pd.DataFrame({
        "sector": scores["sector"],
        "ret": scores["mom_1m"],
        "mcap": fund["marketCap"].reindex(scores.index) if "marketCap" in fund else np.nan,
    }).dropna()
    df = df[df["mcap"] > 0]
    if df.empty:
        return None

    W, H = 1600, 900
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="#1e222d")
    ax = fig.add_axes([0, 0, 1, 0.94])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    sec_sizes = df.groupby("sector")["mcap"].sum().sort_values(ascending=False)
    sec_rects = squarify.squarify(squarify.normalize_sizes(sec_sizes.values, W, H), 0, 0, W, H)
    for sector, sr in zip(sec_sizes.index, sec_rects):
        g = df[df["sector"] == sector].sort_values("mcap", ascending=False)
        head = 22 if sr["dy"] > 60 else 0
        x, y, w, h = sr["x"] + 2, sr["y"] + 2 + head, sr["dx"] - 4, sr["dy"] - 4 - head
        if w <= 0 or h <= 0:
            continue
        rects = squarify.squarify(squarify.normalize_sizes(g["mcap"].values, w, h), x, y, w, h)
        for (t, row), r in zip(g.iterrows(), rects):
            c = CMAP((np.clip(row["ret"], -CLIP, CLIP) + CLIP) / (2 * CLIP))
            ax.add_patch(plt.Rectangle((r["x"], r["y"]), r["dx"], r["dy"], facecolor=c,
                                       edgecolor="#1e222d", linewidth=1))
            if r["dx"] > 34 and r["dy"] > 22:
                fs = max(6, min(16, r["dx"] / 6, r["dy"] / 3))
                ax.text(r["x"] + r["dx"] / 2, r["y"] + r["dy"] / 2, f"{t}\n{row['ret'] * 100:+.1f}%",
                        ha="center", va="center", color="white", fontsize=fs, fontweight="bold")
        if head:
            avg = np.average(g["ret"], weights=g["mcap"])
            ax.text(sr["x"] + 6, sr["y"] + 4, f"{sector}  {avg * 100:+.1f}%", ha="left", va="top",
                    color="#d1d4dc", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.975, f"S&P 500 + Nasdaq-100 · 1-month return · close {asof:%d %b %Y}",
             color="white", fontsize=15, fontweight="bold", va="center")
    fig.text(0.99, 0.975, "size = market cap · green up / red down (±15% = full colour)",
             color="#9598a1", fontsize=10, ha="right", va="center")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
