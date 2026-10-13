"""Figures from a Results table. Colours follow the dataviz method: a fixed categorical order bound to the
POLICY (never its rank), one-hue sequential ramp for magnitude, thin marks, recessive grid, text in ink tokens.
The 7-slot light palette validates (adjacent CVD dE 9.1, normal-vision dE 19.6); three slots are below 3:1
contrast on the surface, so every figure ships with a companion table (`write_report`)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
POLICY_ORDER = ["ucb1", "etc", "glrucb", "swucb", "rawucb", "sswucb", "greedy"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential, light->dark
REGIONS = ["attention-dominant", "contested", "time-dominant"]


def policy_colors(df):
    """Label -> colour, fixed by policy family order (computed from the FULL table so filtering never repaints)."""
    def key(label):
        base = label.split("(")[0]
        return (POLICY_ORDER.index(base) if base in POLICY_ORDER else len(POLICY_ORDER), label)
    labels = sorted(df["policy"].unique(), key=key)
    if len(labels) > len(SLOTS):
        raise ValueError("more than 7 policies: facet or fold into 'Other' instead of cycling colours")
    return {lab: SLOTS[i] for i, lab in enumerate(labels)}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)


def _legend(fig, colors, y=0.0):
    handles = [Line2D([0], [0], color=c, lw=2, marker="o", markersize=6, markeredgecolor=SURFACE, label=l)
               for l, c in colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=8,
               labelcolor=INK2, bbox_to_anchor=(0.5, y))


def regret_grid(df_wide, results, path):
    """Small multiples: columns = region, rows = frontier size m; mean regret vs T, log-log, with T^1 and T^0.5 guides."""
    colors = policy_colors(df_wide)
    long = results.to_long()
    ms = sorted(df_wide["m"].unique())
    fig, axes = plt.subplots(len(ms), 3, figsize=(10.5, 2.6 * len(ms) + 0.9), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, m in enumerate(ms):
        for j, region in enumerate(REGIONS):
            ax = axes[i][j]; _style(ax)
            sub = long[(long["m"] == m) & (long["region"] == region)]
            if sub.empty:
                ax.set_visible(False); continue
            for lab, c in colors.items():
                g = sub[sub["policy"] == lab].groupby("T")["regret_mean"].mean()
                if g.empty:
                    continue
                ax.plot(g.index, np.maximum(g.values, 1e-3), color=c, lw=2, marker="o", markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.0)
            T = np.sort(sub["T"].unique()); ref = sub.groupby("T")["regret_mean"].mean().max()
            for p, ls in ((1.0, "--"), (0.5, ":")):
                ax.plot(T, ref * (T / T.max()) ** p, color=INK2, lw=0.9, ls=ls, alpha=0.7)
            ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0: ax.set_title(region, fontsize=9, color=INK)
            if j == 0: ax.set_ylabel(f"m = {m}\nregret", fontsize=8, color=INK2)
            if i == len(ms) - 1: ax.set_xlabel("horizon T", fontsize=8, color=INK2)
    _legend(fig, colors, y=0.0)
    fig.text(0.995, 0.012, "dashed: slope 1, dotted: slope 1/2", ha="right", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)


def exponent_heatmaps(df, policy_label, path, vmax=1.1):
    """Mean fitted regret exponent over (alpha, beta), one panel per m; dashed diagonal = contested region."""
    sub = df[df["policy"] == policy_label]
    ms = sorted(sub["m"].unique())
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, axes = plt.subplots(1, len(ms), figsize=(3.6 * len(ms) + 1.2, 3.8), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    im = None
    for ax, m in zip(axes[0], ms):
        p = sub[sub["m"] == m].pivot_table(index="alpha", columns="beta", values="exponent", aggfunc="mean")
        p = p.sort_index().sort_index(axis=1)
        im = ax.imshow(p.values, origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(p.columns))); ax.set_xticklabels([f"{x:g}" for x in p.columns], fontsize=7, color=INK2)
        ax.set_yticks(range(len(p.index))); ax.set_yticklabels([f"{x:g}" for x in p.index], fontsize=7, color=INK2)
        ax.plot([-0.5, len(p.columns) - 0.5], [-0.5, len(p.index) - 0.5], color=INK, lw=1, ls="--")
        ax.set_title(f"m = {m}", fontsize=9, color=INK); ax.set_xlabel("beta", fontsize=8, color=INK2)
        for s in ax.spines.values(): s.set_visible(False)
    axes[0][0].set_ylabel("alpha", fontsize=8, color=INK2)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cb.set_label("fitted regret exponent", fontsize=8, color=INK2); cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"{policy_label}   (above the dashed diagonal: attention-dominant; below: time-dominant)",
                 fontsize=9, color=INK, x=0.02, ha="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def region_summary(df, path):
    """Median fitted exponent (with inter-quartile whiskers) per region and frontier size m, one dot per policy."""
    colors = policy_colors(df)
    ms = sorted(df["m"].unique()); labs = list(colors)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True); fig.patch.set_facecolor(SURFACE)
    for ax, region in zip(axes, REGIONS):
        _style(ax); sub = df[df["region"] == region]
        for k, lab in enumerate(labs):
            off = (k - (len(labs) - 1) / 2) * 0.09
            for x, m in enumerate(ms):
                v = sub[(sub["policy"] == lab) & (sub["m"] == m)]["exponent"].dropna()
                if v.empty: continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([x + off] * 2, [q1, q3], color=colors[lab], lw=2, solid_capstyle="round")
                ax.plot(x + off, med, "o", color=colors[lab], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([f"m={m}" for m in ms], fontsize=8, color=INK2)
        ax.set_title(region, fontsize=9, color=INK); ax.set_xlim(-0.6, len(ms) - 0.4)
    axes[0].set_ylabel("fitted regret exponent", fontsize=8, color=INK2)
    _legend(fig, colors, y=0.0); fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)


def write_report(results, outdir):
    """All figures plus the companion tables (the 'table view' required when contrast is below 3:1)."""
    import os
    os.makedirs(outdir, exist_ok=True)
    df = results.df; paths = []
    p = os.path.join(outdir, "regret_grid.png"); regret_grid(df, results, p); paths.append(p)
    p = os.path.join(outdir, "region_summary.png"); region_summary(df, p); paths.append(p)
    for lab in sorted(df["policy"].unique()):
        safe = "".join(ch if ch.isalnum() else "_" for ch in lab)
        p = os.path.join(outdir, f"exponent_heatmap_{safe}.png"); exponent_heatmaps(df, lab, p); paths.append(p)
    tab = results.exponents(); p = os.path.join(outdir, "exponents_by_region_m_policy.csv"); tab.to_csv(p, index=False)
    paths.append(p)
    return paths
