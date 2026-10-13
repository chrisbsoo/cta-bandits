"""The (alpha, beta) parameter grid: one small cell per (alpha, beta), RdYlGn_r colour = fitted regret exponent.
Inside each cell a strip runs over the frontier size m = 1..K (left to right), averaged over random instances.

    python -m tacb.plots_grid results.csv --out figures/
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .experiment import Results


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def ab_grid(df, policy_label, path, cmap="RdYlGn_r", interpolation="nearest", vmin=None, vmax=None):
    d = df[df["policy"] == policy_label]
    alphas, betas, ms = sorted(d["alpha"].unique()), sorted(d["beta"].unique()), sorted(d["m"].unique())
    A = np.full((len(alphas), len(betas), len(ms)), np.nan)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for k, m in enumerate(ms):
                x = d[(d.alpha == a) & (d.beta == b) & (d.m == m)]["exponent"].to_numpy(float)
                if np.isfinite(x).any():
                    A[i, j, k] = np.nanmean(x)                  # average case over instances
    vmin = np.nanpercentile(A, 5) if vmin is None else vmin
    vmax = np.nanpercentile(A, 95) if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    na, nb = len(alphas), len(betas)
    fig = plt.figure(figsize=(nb * 1.9 + 1.2, na * 1.9 + 1.0), constrained_layout=True)
    gs = fig.add_gridspec(na, nb + 1, width_ratios=[1] * nb + [0.08])
    for i in range(na):
        for j in range(nb):
            ax = fig.add_subplot(gs[na - 1 - i, j])             # alpha increases upward
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#999999"); sp.set_linewidth(0.8)
            if np.isnan(A[i, j]).all():
                ax.set_facecolor("#e8e8e8"); continue
            ax.imshow(A[i, j][None, :], aspect="auto", cmap=cmap, norm=norm,
                      extent=[ms[0] - .5, ms[-1] + .5, 0, 1], interpolation=interpolation)
            if abs(alphas[i] - betas[j]) < 1e-9:                # contested diagonal
                for sp in ax.spines.values():
                    sp.set_edgecolor("#0b0b0b"); sp.set_linewidth(1.6)
            if j == 0:
                ax.set_ylabel(f"{alphas[i]:.2f}", fontsize=11, rotation=0, ha="right", va="center", labelpad=8)
            if i == 0:
                ax.set_xlabel(f"{betas[j]:.2f}", fontsize=11, labelpad=8)
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("fitted regret exponent  (Regret $= \\Theta(T^{\\,\\mathrm{exponent}})$)",
                 fontsize=11, rotation=270, labelpad=20)
    fig.suptitle(f"{policy_label}: fitted regret exponent per $(\\alpha,\\beta)$ cell\n"
                 f"each cell sweeps frontier size $m={ms[0]}\\rightarrow{ms[-1]}$ (left to right)\n"
                 f"mean over instances; bold frame = contested diagonal", fontsize=12, fontweight="bold")
    fig.supxlabel(r"$\beta \rightarrow$", fontsize=15)
    fig.supylabel(r"$\alpha \rightarrow$", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _cells(df, policy_label, ms, value_fn):
    """Tensor (alpha, beta, m) of the instance-mean of value_fn(row), plus the std-error over instances."""
    d = df[df["policy"] == policy_label]
    alphas, betas = sorted(d["alpha"].unique()), sorted(d["beta"].unique())
    ms = sorted(d["m"].unique()) if ms is None else list(ms)
    M = np.full((len(alphas), len(betas), len(ms)), np.nan)
    SE = np.full_like(M, np.nan)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for k, m in enumerate(ms):
                x = np.array([value_fn(r) for _, r in d[(d.alpha == a) & (d.beta == b) & (d.m == m)].iterrows()], float)
                x = x[np.isfinite(x)]
                if len(x):
                    M[i, j, k] = x.mean()
                    SE[i, j, k] = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return alphas, betas, ms, M, SE


def _draw(alphas, betas, ms, M, mask_grey, mask_hatch, vmin, vmax, cmap, cblabel, title, path):
    na, nb = len(alphas), len(betas)
    fig = plt.figure(figsize=(nb * 2.1 + 1.4, na * 1.9 + 1.6), constrained_layout=True)
    gs = fig.add_gridspec(na, nb + 1, width_ratios=[1] * nb + [0.08])
    norm = plt.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap).copy()
    for i in range(na):
        for j in range(nb):
            ax = fig.add_subplot(gs[na - 1 - i, j])
            for sp in ax.spines.values():
                sp.set_edgecolor("#999999"); sp.set_linewidth(0.8)
            if abs(alphas[i] - betas[j]) < 1e-9:
                for sp in ax.spines.values():
                    sp.set_edgecolor("#0b0b0b"); sp.set_linewidth(1.6)
            ax.set_yticks([])
            ax.set_xticks(np.arange(len(ms)))
            ax.set_xlim(-0.5, len(ms) - 0.5)
            if i == 0:
                ax.set_xticklabels([str(m) for m in ms], fontsize=7)
                ax.set_xlabel(f"$\\beta$={betas[j]:.2f}", fontsize=10, labelpad=4)
            else:
                ax.set_xticklabels([])
            ax.tick_params(length=2)
            for k in range(len(ms)):
                v = M[i, j, k]
                if not np.isfinite(v):
                    face = "#e8e8e8"
                elif mask_grey[i, j, k]:
                    face = "#d9d9d9"
                else:
                    face = cm(norm(v))
                ax.add_patch(plt.Rectangle((k - 0.5, 0), 1, 1, facecolor=face, edgecolor="#444444", linewidth=0,
                                           hatch="////" if (np.isfinite(v) and mask_hatch[i, j, k]) else None))
            ax.set_ylim(0, 1)
            if j == 0:
                ax.set_ylabel(f"$\\alpha$={alphas[i]:.2f}", fontsize=10, rotation=0, ha="right", va="center", labelpad=8)
    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), cax=cax)
    cb.set_label(cblabel, fontsize=10, rotation=270, labelpad=18)
    fig.suptitle(title, fontsize=10.5, fontweight="bold")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def rel_regret_grid(df, policy_label, path, ms=None, vmax=0.5, cmap="RdYlGn_r"):
    """Colour = relative regret at the largest horizon, regret / V*(T): 0 = near-optimal, larger = more is lost.
    No curve fit. Fixed colour scale shared by every policy. Sub-columns are frontier sizes m (tick labels)."""
    f = lambda r: r["regret_mean"][-1] / r["v_star"]
    alphas, betas, ms, M, SE = _cells(df, policy_label, ms, f)
    T = int(df["horizons"].iloc[0][-1]); runs = int(df["n_runs"].iloc[0]); ninst = df["inst_seed"].nunique()
    title = (f"{policy_label}: relative regret at T={T}  [regret / V*(T)]\n"
             f"sub-columns = frontier size m = {', '.join(str(m) for m in ms)}\n"
             f"mean over {ninst} instances x {runs} runs; bold frame = contested diagonal")
    return _draw(alphas, betas, ms, M, np.zeros_like(M, bool), np.zeros_like(M, bool), 0.0, vmax, cmap,
                 "relative regret at T_max", title, path)


def exponent_grid(df, policy_label, path, ms=None, cmap="RdYlGn_r", se_max=0.15, rel_min=0.02):
    """Fitted exponent, shown only where it means something: grey = regret is already tiny (<rel_min of V*), so the
    exponent is not informative; hatched = standard error over instances above se_max (unreliable)."""
    alphas, betas, ms, M, SE = _cells(df, policy_label, ms, lambda r: r["exponent"])
    _, _, _, REL, _ = _cells(df, policy_label, ms, lambda r: r["regret_mean"][-1] / r["v_star"])
    grey = REL < rel_min
    hatch = (SE > se_max) | ~np.isfinite(SE)
    h = df["horizons"].iloc[0]
    title = (f"{policy_label}: fitted regret exponent  (T={h[0]}..{h[-1]}, {len(h)} points)\n"
             f"grey = regret < {rel_min:.0%} of V* (uninformative)\n"
             f"hatched = s.e. over instances > {se_max}; bold frame = contested diagonal")
    return _draw(alphas, betas, ms, M, grey, hatch, 0.0, 1.0, cmap,
                 "fitted regret exponent", title, path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("csv"); p.add_argument("--out", default="figures")
    p.add_argument("--cmap", default="RdYlGn_r"); p.add_argument("--interp", default="nearest")
    a = p.parse_args(argv)
    import os
    os.makedirs(a.out, exist_ok=True)
    df = Results.from_csv(a.csv).df
    for lab in df["policy"].unique():
        if "v_star" in df.columns:           # new-format results
            print("wrote", rel_regret_grid(df, lab, os.path.join(a.out, f"rel_regret_{_slug(lab)}.png"), cmap=a.cmap))
            print("wrote", exponent_grid(df, lab, os.path.join(a.out, f"exponent_{_slug(lab)}.png"), cmap=a.cmap))
        else:
            print("wrote", ab_grid(df, lab, os.path.join(a.out, f"ab_grid_{_slug(lab)}.png"), a.cmap, a.interp))


if __name__ == "__main__":
    main()