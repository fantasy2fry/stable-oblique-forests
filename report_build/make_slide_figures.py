"""Figures sized for the presentation slides (report/presentation.tex).

Like make_figures.py, this reads SAVED results only and fits no model. Figures are drawn
at the slide text width (5.5 in for the 16:9 Beamer frame), so they are included without
rescaling and their 8-9 pt labels match the slide text. Run make_tables.py first (the
heatmap reads report_build/tables/synthetic_primary_tests.csv).

    python report_build/make_slide_figures.py
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
W = 5.5  # inches, text width of the 16:9 slides

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "mathtext.fontset": "dejavusans",
})

FUNC_MATH = {"axis": r"$f_{\mathrm{axis}}$", "oblique": r"$f_{\mathrm{oblique}}$",
             "oblique2": r"$f_{\mathrm{oblique2}}$", "mixed": r"$f_{\mathrm{mixed}}$"}
FUNCS = ["axis", "oblique", "oblique2", "mixed"]
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9]
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


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"wrote report/figures/{name}")


def legend_axis(ax, handles):
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(-0.05, 0.5),
              handlelength=2.2, labelspacing=0.9)


def model_handles(models):
    return [Line2D([0], [0], color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                   marker=STYLE[m]["marker"], markersize=4.5, label=STYLE[m]["label"])
            for m in models]


# ---------------------------------------------------------------- excess Brier (2 functions)
raw = pd.read_csv(OUT / "correlation" / "results_raw.csv")
raw["excess"] = raw["brier"] - raw["oracle_brier"]
agg = raw.groupby(["function", "rho", "model"])["excess"].agg(["mean", "std", "count"]).reset_index()
agg["se"] = agg["std"] / np.sqrt(agg["count"])
fig = plt.figure(figsize=(W, 1.95), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.72])
for k, f in enumerate(["axis", "oblique2"]):
    ax = fig.add_subplot(gs[0, k])
    for m in MAIN:
        d = agg[(agg["function"] == f) & (agg["model"] == m)].sort_values("rho")
        s = STYLE[m]
        ax.errorbar(d["rho"], 1e3 * d["mean"], yerr=1e3 * d["se"], color=s["color"], ls=s["ls"],
                    marker=s["marker"], markersize=3.8, capsize=1.8, elinewidth=0.7,
                    markeredgewidth=0.7)
    ax.set_title(FUNC_MATH[f])
    ax.set_xticks(RHOS, ["0", ".25", ".5", ".75", ".9"])
    ax.set_xlabel(r"correlation $\rho$")
    if k == 0:
        ax.set_ylabel(r"excess Brier $\times10^{3}$")
legend_axis(fig.add_subplot(gs[0, 2]), model_handles(MAIN))
save(fig, "slide_excess_brier.pdf")

# ---------------------------------------------------------------- heatmap of paired differences
prim = pd.read_csv(TAB / "synthetic_primary_tests.csv")
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a29f", "#b2302f"])
panels = [(L0, "vs. L0 only"), (UNIF, "vs. uniform control"), (SHUF, "vs. shuffled control")]
fig, axes = plt.subplots(1, 3, figsize=(W, 1.75), constrained_layout=True)
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
            ax.text(j, i, f"{v:+.0f}" if abs(v) >= 9.95 else f"{v:+.1f}",
                    ha="center", va="center", fontsize=6.6,
                    fontweight="bold" if sig[i, j] else "normal",
                    color="white" if abs(v) > 0.62 * lim else INK)
    ax.set_xticks(range(len(RHOS)), ["0", ".25", ".5", ".75", ".9"])
    ax.set_yticks(range(len(FUNCS)), [FUNC_MATH[f] for f in FUNCS] if comp == L0 else [])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(r"$\rho$", labelpad=1)
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
save(fig, "slide_heatmap.pdf")

# ---------------------------------------------------------------- regularization paths
fam_style = {
    "L0": dict(color=ORANGE, marker="v", mfc=ORANGE, label=r"L0 only ($\gamma=0$)"),
    "Stability": dict(color=BLUE, marker="o", mfc=BLUE, label="feature stability"),
    "Uniform": dict(color=INK2, marker="o", mfc="none", label="uniform control"),
    "Shuffled": dict(color=INK2, marker="^", mfc="none", label="shuffled control"),
}
fig = plt.figure(figsize=(W, 2.0), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.72])
for k, f in enumerate(["axis", "oblique2"]):
    ax = fig.add_subplot(gs[0, k])
    reg = pd.read_csv(OUT / "regularization" / f / "summary.csv")
    for fam in ["Uniform", "Shuffled", "Stability", "L0"]:
        g = reg[reg["model"] == fam].sort_values("nodes_per_tree_mean")
        s = fam_style[fam]
        ax.plot(g["nodes_per_tree_mean"], 1e3 * g["brier_mean"], ls="-" if fam == "L0" else "none",
                color=s["color"], marker=s["marker"], mfc=s["mfc"], mec=s["color"],
                markersize=4.3, markeredgewidth=0.8, lw=1.2,
                zorder=3 if fam in ("L0", "Stability") else 2)
    base = reg[(reg["model"] == "L0") & (reg["sparsity_penalty"] == 0)].iloc[0]
    ax.plot(base["nodes_per_tree_mean"], 1e3 * base["brier_mean"], marker="*", color=INK,
            markersize=8.5, ls="none", zorder=4)
    ax.set_title(FUNC_MATH[f] + r", $\rho=0.9$")
    ax.set_xlabel("nodes per tree")
    if k == 0:
        ax.set_ylabel(r"test Brier $\times10^{3}$")
handles = [Line2D([0], [0], color=s["color"], marker=s["marker"], mfc=s["mfc"], mec=s["color"],
                  ls="-" if fam == "L0" else "none", markersize=5, label=s["label"])
           for fam, s in fam_style.items()]
handles.append(Line2D([0], [0], color=INK, marker="*", ls="none", markersize=8,
                      label="unpenalized"))
legend_axis(fig.add_subplot(gs[0, 2]), handles)
save(fig, "slide_regularization_paths.pdf")

# ---------------------------------------------------------------- OpenML primary comparison
res = pd.read_csv(OUT / "openml" / "tables" / "openml_benchmark_results.csv")
wide = res.pivot_table(index=["dataset", "repeat", "fold"], columns="model", values="brier")
diff = 1e3 * (wide["Oblique L0+stability tuned"] - wide["Oblique L0 tuned"])
order = diff.groupby("dataset").mean().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(3.05, 2.75), constrained_layout=True)
for i, d in enumerate(order):
    vals = diff.xs(d, level="dataset").to_numpy()
    ax.plot(vals, np.full(len(vals), i), ls="none", marker="o", markersize=2.6, color=MUTED,
            alpha=0.8, markeredgewidth=0)
    mean = vals.mean()
    ax.plot(mean, i, ls="none", marker="D", markersize=4.6, color=BLUE if mean < 0 else "#b2302f")
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(order)), order, fontsize=7)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel(r"$\Delta$ Brier $\times10^{3}$ (L0+stab. $-$ L0)")
top = ax.get_xaxis_transform()  # x in data units, y in axes units
ax.text(-0.25, 1.01, "← better", transform=top, fontsize=7.5, color=BLUE,
        ha="right", va="bottom")
ax.text(0.25, 1.01, "worse →", transform=top, fontsize=7.5, color="#b2302f",
        ha="left", va="bottom")
save(fig, "slide_openml_primary.pdf")

# ---------------------------------------------------------------- 2-D demo: four panels
# Panel boxes (pixels) of report/figures/demo_mixed_maps.png, located from its 2x3 layout.
img = plt.imread(FIG / "demo_mixed_maps.png")[..., :3]
ROWS = [(86, 680), (768, 1362)]
COLS = [(86, 754), (812, 1479), (1537, 2204)]
crops = [
    ("CART tree", ROWS[0], COLS[0]),
    ("Random forest", ROWS[0], COLS[1]),
    ("Oblique forest (ours)", ROWS[0], COLS[2]),
    ("True probability", ROWS[1], COLS[2]),
]
fig, axes = plt.subplots(2, 2, figsize=(2.75, 2.85), constrained_layout=True)
for ax, (title, (r0, r1), (c0, c1)) in zip(axes.ravel(), crops):
    ax.imshow(img[r0:r1 + 1, c0:c1 + 1], interpolation="lanczos")
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(MUTED)
        spine.set_linewidth(0.6)
fig.savefig(FIG / "slide_demo_mixed.png", dpi=300)
plt.close(fig)
print("wrote report/figures/slide_demo_mixed.png")

# ---------------------------------------------------------------- 2-D demos: three more datasets
# Rows: w, diagonal, moons. Columns: CART, random forest, unpenalized oblique forest, truth.
# Panel boxes (pixels) of report/figures/demo_<name>_maps.png (same 2x3 layout as above).
DEMO_ROWS = [(86, 680), (768, 1362)]
DEMO_COLS = {"w": [(107, 768), (826, 1486), (1544, 2205)],
             "diagonal": [(86, 754), (812, 1479), (1537, 2204)],
             "moons": [(107, 768), (826, 1486), (1544, 2205)]}
titles = ["CART", "Random forest", "Oblique (ours)", "Truth"]
fig, axes = plt.subplots(3, 4, figsize=(3.55, 2.85), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.02)
for i, name in enumerate(["w", "diagonal", "moons"]):
    img = plt.imread(FIG / f"demo_{name}_maps.png")[..., :3]
    cols = DEMO_COLS[name]
    boxes = [(DEMO_ROWS[0], cols[0]), (DEMO_ROWS[0], cols[1]), (DEMO_ROWS[0], cols[2]),
             (DEMO_ROWS[1], cols[2]) if name != "moons" else None]
    for j, box in enumerate(boxes):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if box is None:
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.text(0.5, 0.5, "not\navailable", ha="center", va="center", fontsize=6.5,
                    color=MUTED, transform=ax.transAxes)
        else:
            (r0, r1), (c0, c1) = box
            ax.imshow(img[r0:r1 + 1, c0:c1 + 1], interpolation="lanczos", aspect="auto")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(MUTED)
                spine.set_linewidth(0.5)
        if i == 0:
            ax.set_title(titles[j], fontsize=6.5, pad=2)
        if j == 0:
            ax.set_ylabel(name, fontsize=8, color=INK, labelpad=3)
fig.savefig(FIG / "slide_demo_more.png", dpi=300)
plt.close(fig)
print("wrote report/figures/slide_demo_more.png")

# ---------------------------------------------------------------- structure vs rho (slide 9)
summ = pd.read_csv(OUT / "correlation" / "summary.csv")
chance = {f: RELEVANT_N / 20 for f, RELEVANT_N in {"axis": 1, "oblique": 3, "oblique2": 2,
                                                   "mixed": 4}.items()}
summ["rel_excess"] = summ["relevant_fraction_mean"] - summ["function"].map(chance)
STYLE_UNIF = dict(label="Uniform control", color=INK2, ls="none", marker="o", mfc="none")
struct_models = [OBL, L0, STAB, FULL]
fig, axes = plt.subplots(2, 1, figsize=(2.45, 2.85), sharex=True, constrained_layout=True)
for ax, key, label in [(axes[0], "rel_excess", "relevant share\nabove chance"),
                       (axes[1], "nodes_per_tree_mean", "nodes per tree")]:
    for m in struct_models + [UNIF]:
        d = summ[summ["model"] == m].groupby("rho")[key].mean()
        if m == UNIF:
            ax.plot(d.index, d.values, ls="none", marker="o", mfc="none", mec=INK2,
                    markersize=5.5, markeredgewidth=0.8, zorder=4)
        else:
            s = STYLE[m]
            ax.plot(d.index, d.values, color=s["color"], ls=s["ls"], marker=s["marker"],
                    markersize=3.6)
    ax.set_ylabel(label, fontsize=7.5)
axes[1].set_xticks(RHOS, ["0", ".25", ".5", ".75", ".9"])
axes[1].set_xlabel(r"correlation $\rho$")
handles = model_handles(struct_models) + [
    Line2D([0], [0], ls="none", marker="o", mfc="none", mec=INK2, markersize=5.5,
           label="Uniform control")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
           fontsize=6.5, handlelength=1.8, columnspacing=0.8)
save(fig, "slide_structure.pdf")

# ---------------------------------------------------------------- OpenML: RF default vs tuned L0 (backup)
diff_rf = 1e3 * (wide["RF default"] - wide["Oblique L0 tuned"])
order_rf = diff_rf.groupby("dataset").mean().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(4.2, 2.6), constrained_layout=True)
for i, d in enumerate(order_rf):
    vals = diff_rf.xs(d, level="dataset").to_numpy()
    ax.plot(vals, np.full(len(vals), i), ls="none", marker="o", markersize=2.6, color=MUTED,
            alpha=0.8, markeredgewidth=0)
    mean = vals.mean()
    ax.plot(mean, i, ls="none", marker="D", markersize=4.6, color=INK2 if mean < 0 else BLUE)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(order_rf)), order_rf, fontsize=7)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel(r"$\Delta$ Brier $\times10^{3}$ (RF default $-$ L0 tuned)")
top = ax.get_xaxis_transform()
ax.text(-5, 1.01, "\u2190 RF better", transform=top, fontsize=7.5, color=INK2, ha="right",
        va="bottom")
ax.text(5, 1.01, "oblique better \u2192", transform=top, fontsize=7.5, color=BLUE, ha="left",
        va="bottom")
save(fig, "slide_openml_rf.pdf")
