"""Summary figures for the report, drawn from SAVED experiment outputs only.

Nothing here fits a model. The script reads the CSV files of the final run
(``outputs/final``) and ``report_build/tables/synthetic_primary_tests.csv`` (written by
``make_tables.py``, so run that first) and writes vector PDFs to ``report/figures``.
Figures are drawn at the report's text width (6.3 in), so they are included at
``\\linewidth`` without rescaling and the 7-8 pt labels stay readable.

    python report_build/make_tables.py
    python report_build/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final"
TAB = ROOT / "report_build" / "tables"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

WIDTH = 6.3  # inches, = \linewidth of the report

# ---- colour tokens: three method variants in a CVD-checked categorical palette,
# ---- the two reference forests in neutral greys (secondary encoding: markers + dashes)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.3, "lines.markersize": 4, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "mathtext.fontset": "dejavusans",
})

FUNCS = ["axis", "oblique", "oblique2", "mixed"]
FUNC_MATH = {"axis": r"$f_{\mathrm{axis}}$", "oblique": r"$f_{\mathrm{oblique}}$",
             "oblique2": r"$f_{\mathrm{oblique2}}$", "mixed": r"$f_{\mathrm{mixed}}$"}
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9]
RELEVANT = {"axis": 1, "oblique": 3, "oblique2": 2, "mixed": 4}

RF, OBL, L0, STAB, FULL = (
    "Random forest", "Oblique", "Oblique + L0", "Oblique + stability", "Oblique + L0 + stability",
)
UNIF, SHUF = "Oblique + L0 + uniform stability", "Oblique + L0 + shuffled stability"
STYLE = {
    RF: dict(label="Random forest", color=MUTED, ls=":", marker="^"),
    OBL: dict(label="Unpenalized oblique", color=INK2, ls="--", marker="s"),
    L0: dict(label="L0 only", color=ORANGE, ls="-", marker="v"),
    STAB: dict(label="Stability only", color=AQUA, ls="-", marker="D"),
    FULL: dict(label="L0 + stability (full)", color=BLUE, ls="-", marker="o"),
}
MAIN = [RF, OBL, L0, STAB, FULL]


def line(ax, x, y, err, model, **extra):
    s = STYLE[model]
    ax.errorbar(x, y, yerr=err, color=s["color"], ls=s["ls"], marker=s["marker"],
                label=s["label"], capsize=1.8, elinewidth=0.7, markeredgewidth=0.7,
                markersize=3.6 if s["marker"] != "D" else 3.2, **extra)


def legend_top(fig, models, ncol=5, y=1.0):
    handles = [Line2D([0], [0], color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                      marker=STYLE[m]["marker"], markersize=4, label=STYLE[m]["label"])
               for m in models]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               handlelength=2.4, columnspacing=1.4)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"wrote report/figures/{name}")


raw = pd.read_csv(OUT / "correlation" / "results_raw.csv")
summ = pd.read_csv(OUT / "correlation" / "summary.csv")

# ============================================== 1. synthetic: excess Brier vs rho
raw["excess"] = raw["brier"] - raw["oracle_brier"]
agg = raw.groupby(["function", "rho", "model"])["excess"].agg(["mean", "std", "count"]).reset_index()
agg["se"] = agg["std"] / np.sqrt(agg["count"])
fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.3), sharex=True, constrained_layout=True)
for ax, f in zip(axes.ravel(), FUNCS):
    for m in MAIN:
        d = agg[(agg["function"] == f) & (agg["model"] == m)].sort_values("rho")
        line(ax, d["rho"], 1e3 * d["mean"], 1e3 * d["se"], m)
    ax.set_title(FUNC_MATH[f])
    ax.set_xticks(RHOS, [f"{r:g}" for r in RHOS])
for ax in axes[1]:
    ax.set_xlabel(r"correlation $\rho$ within $x_1,\dots,x_4$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"excess Brier $\times 10^{3}$")
legend_top(fig, MAIN, y=1.0)
save(fig, "fig_synthetic_excess_brier.pdf")

# ============================================== 2. synthetic: heatmap of paired differences
prim = pd.read_csv(TAB / "synthetic_primary_tests.csv")
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a29f", "#b2302f"])
panels = [(L0, "vs. L0 only"), (UNIF, "vs. uniform control"), (SHUF, "vs. shuffled control")]
fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.05), constrained_layout=True)
for ax, (comp, title) in zip(axes, panels):
    t = prim[prim["competitor"] == comp]
    cell = {(r.function, round(r.rho, 2)): r for r in t.itertuples()}
    grid = np.array([[1e4 * cell[(f, r)].mean_delta_brier for r in RHOS] for f in FUNCS])
    sig = np.array([[cell[(f, r)].p_holm_global < 0.05 for r in RHOS] for f in FUNCS])
    lim = float(np.abs(grid).max())
    im = ax.imshow(grid, cmap=DIVERGING, norm=TwoSlopeNorm(0.0, -lim, lim), aspect="auto")
    for i in range(len(FUNCS)):
        for j in range(len(RHOS)):
            v = grid[i, j]
            ax.text(j, i, f"{v:+.1f}" + ("*" if sig[i, j] else ""), ha="center", va="center",
                    fontsize=6.3, color="white" if abs(v) > 0.62 * lim else INK)
    ax.set_xticks(range(len(RHOS)), [f"{r:g}" for r in RHOS])
    ax.set_yticks(range(len(FUNCS)), [FUNC_MATH[f] for f in FUNCS] if comp == L0 else [])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(r"$\rho$")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)
save(fig, "fig_synthetic_heatmap.pdf")

# ============================================== 3. synthetic: structure and selection
metrics = [("relevant_fraction", "relevant share of\nsplit features"),
           ("mean_support_size", r"support size $\|w\|_0$"),
           ("nodes_per_tree", "nodes per tree")]
fig, axes = plt.subplots(4, 3, figsize=(WIDTH, 6.0), sharex=True, constrained_layout=True)
for i, f in enumerate(FUNCS):
    for j, (key, label) in enumerate(metrics):
        ax = axes[i, j]
        models = MAIN if key == "nodes_per_tree" else [OBL, L0, STAB, FULL]
        for m in models:
            d = summ[(summ["function"] == f) & (summ["model"] == m)].sort_values("rho")
            se = d[f"{key}_std"] / np.sqrt(d["n_replicates"])
            line(ax, d["rho"], d[f"{key}_mean"], se, m)
        if key == "relevant_fraction":
            ax.axhline(RELEVANT[f] / 20, color=MUTED, lw=0.8, ls=(0, (1, 2)))
            ax.text(0.02, RELEVANT[f] / 20, " chance", color=INK2, fontsize=6.3, va="bottom")
        ax.set_xticks(RHOS, [f"{r:g}" for r in RHOS])
        if i == 0:
            ax.set_title(label.replace("\n", " "))
        if j == 0:
            ax.set_ylabel(FUNC_MATH[f], rotation=0, ha="right", va="center", fontsize=8.5,
                          color=INK)
        if i == len(FUNCS) - 1:
            ax.set_xlabel(r"$\rho$")
legend_top(fig, MAIN, y=1.0)
save(fig, "fig_synthetic_structure.pdf")

# ============================================== 4. regularization paths
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75), constrained_layout=True)
fam_style = {
    "L0": dict(color=ORANGE, marker="v", mfc=ORANGE, label=r"L0 only ($\gamma=0$)"),
    "Stability": dict(color=BLUE, marker="o", mfc=BLUE, label="feature stability"),
    "Uniform": dict(color=INK2, marker="o", mfc="none", label="uniform control"),
    "Shuffled": dict(color=INK2, marker="^", mfc="none", label="shuffled control"),
}
for ax, f in zip(axes, ["axis", "oblique2"]):
    reg = pd.read_csv(OUT / "regularization" / f / "summary.csv")
    for fam in ["Uniform", "Shuffled", "Stability", "L0"]:
        g = reg[reg["model"] == fam].sort_values("nodes_per_tree_mean")
        s = fam_style[fam]
        ax.plot(g["nodes_per_tree_mean"], 1e3 * g["brier_mean"], ls="-" if fam == "L0" else "none",
                color=s["color"], marker=s["marker"], mfc=s["mfc"], mec=s["color"],
                markersize=4.2, markeredgewidth=0.8, lw=1.1, label=s["label"],
                zorder=3 if fam in ("L0", "Stability") else 2)
    base = reg[(reg["model"] == "L0") & (reg["sparsity_penalty"] == 0)].iloc[0]
    ax.plot(base["nodes_per_tree_mean"], 1e3 * base["brier_mean"], marker="*", color=INK,
            markersize=8, ls="none", label=r"unpenalized ($\lambda=\gamma=0$)", zorder=4)
    ax.set_title(FUNC_MATH[f] + r", $\rho=0.9$")
    ax.set_xlabel("mean nodes per tree")
axes[0].set_ylabel(r"test Brier $\times 10^{3}$")
handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index(s["label"]) for s in (fam_style["L0"], fam_style["Stability"],
                                             fam_style["Uniform"], fam_style["Shuffled"])]
order.append(len(labels) - 1)
fig.legend([handles[k] for k in order], [labels[k] for k in order], loc="lower center",
           bbox_to_anchor=(0.5, 1.0), ncol=5, handlelength=1.8, columnspacing=1.2)
save(fig, "fig_regularization_paths.pdf")

# ============================================== 5. OpenML: dataset-level paired differences
res = pd.read_csv(OUT / "openml" / "tables" / "openml_benchmark_results.csv")
wide = res.pivot_table(index=["dataset", "repeat", "fold"], columns="model", values="brier")
diff_a = 1e3 * (wide["Oblique L0+stability tuned"] - wide["Oblique L0 tuned"])
diff_b = 1e3 * (wide["RF default"] - wide["Oblique L0 tuned"])
order = diff_a.groupby("dataset").mean().sort_values().index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5), sharey=True, constrained_layout=True)
specs = [
    (axes[0], diff_a, "L0+stability (tuned) $-$ L0 (tuned)", INK),
    (axes[1], diff_b, "RF default $-$ L0 (tuned)", INK),
]
for ax, diff, title, color in specs:
    for i, d in enumerate(order):
        vals = diff.xs(d, level="dataset").to_numpy()
        ax.plot(vals, np.full(len(vals), i), ls="none", marker="o", markersize=2.6,
                color=MUTED, alpha=0.75, markeredgewidth=0)
        ax.plot(vals.mean(), i, ls="none", marker="D", markersize=4.2, color=color)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title(title)
    ax.set_xlabel(r"$\Delta$ Brier $\times 10^{3}$  (negative: $A$ better)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(order)), order)
axes[0].invert_yaxis()
handles = [Line2D([0], [0], ls="none", marker="D", color=INK, markersize=4.2,
                  label="mean over the 5 outer folds"),
           Line2D([0], [0], ls="none", marker="o", color=MUTED, markersize=2.8,
                  label="single outer fold")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "fig_openml_differences.pdf")

# ============================================== 6. synthetic: other metrics (appendix)
other = [("accuracy", "accuracy"), ("f1", "F1 score"),
         ("average_precision", "average precision"), ("nodes_per_tree", "nodes per tree")]
per_rep = raw.groupby(["rho", "replicate", "model"])[[k for k, _ in other]].mean().reset_index()
fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.1), sharex=True, constrained_layout=True)
for ax, (key, label) in zip(axes.ravel(), other):
    for m in MAIN:
        d = per_rep[per_rep["model"] == m].groupby("rho")[key].agg(["mean", "std", "count"])
        line(ax, d.index, d["mean"], d["std"] / np.sqrt(d["count"]), m)
    ax.set_title(label)
    ax.set_xticks(RHOS, [f"{r:g}" for r in RHOS])
for ax in axes[1]:
    ax.set_xlabel(r"correlation $\rho$ within $x_1,\dots,x_4$")
legend_top(fig, MAIN, y=1.0)
save(fig, "fig_synthetic_other_metrics.pdf")
