"""Derive the report's tables and key numbers from SAVED experiment outputs only.

Nothing here fits a model: the script reads CSV files written by the final experiment
run (``outputs/final``) and writes LaTeX tabular bodies plus ``key_numbers.json`` to
``report_build/tables``. It does not import ``stable_oblique_forests``.

Pipeline (run from the repository root, a few seconds in total):

    python report_build/make_tables.py    # tables + key numbers   -> report_build/tables
    python report_build/make_figures.py   # summary figures (PDF)   -> report/figures
    python report_build/sync_tables.py    # paste tables into report/report_final.tex

Use the project environment, e.g. ``.venv/Scripts/python.exe`` (Windows) or
``.venv/bin/python`` (macOS/Linux), or ``uv run python``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final"
TAB = ROOT / "report_build" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

FUNCS = ["axis", "oblique", "oblique2", "mixed"]
FUNC_TEX = {
    "axis": r"$f_{\mathrm{axis}}$",
    "oblique": r"$f_{\mathrm{oblique}}$",
    "oblique2": r"$f_{\mathrm{oblique2}}$",
    "mixed": r"$f_{\mathrm{mixed}}$",
}
FUNC_PLAIN = {"axis": "f_axis", "oblique": "f_oblique", "oblique2": "f_oblique2",
              "mixed": "f_mixed"}
RELEVANT = {"axis": 1, "oblique": 3, "oblique2": 2, "mixed": 4}
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9]

RF, OBL, L0, STAB, FULL = (
    "Random forest", "Oblique", "Oblique + L0", "Oblique + stability", "Oblique + L0 + stability",
)
UNIF, SHUF, TRAIN = (
    "Oblique + L0 + uniform stability", "Oblique + L0 + shuffled stability",
    "Oblique + training selection",
)
SYN_MODELS = [RF, OBL, L0, STAB, FULL, UNIF, SHUF, TRAIN]
SYN_SHORT = {
    RF: "RF", OBL: "Obl.", L0: "L0", STAB: "Stab.", FULL: "L0+Stab.",
    UNIF: "Unif.", SHUF: "Shuf.", TRAIN: "Train-sel.",
}
SYN_LONG = {
    RF: "Random forest", OBL: "Unpenalized oblique", L0: "L0 only",
    STAB: "Stability only", FULL: "L0 + stability (full)", UNIF: "Uniform control",
    SHUF: "Shuffled control", TRAIN: "Training selection",
}

O_RF, O_RFM, O_BASE, O_L0, O_FULL = (
    "RF default", "RF size-matched", "Oblique baseline", "Oblique L0 tuned",
    "Oblique L0+stability tuned",
)
OML_MODELS = [O_RF, O_RFM, O_BASE, O_L0, O_FULL]
OML_SHORT = {O_RF: "RF", O_RFM: "RF-size", O_BASE: "Obl.", O_L0: "L0 (tuned)",
             O_FULL: "L0+Stab. (tuned)"}

KEY: dict = {}  # key numbers quoted in the text, written to tables/key_numbers.json


# ----------------------------------------------------------------------------- helpers
def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        out[idx] = min(1.0, running)
    return out


def fmt_p(p, bold_below=0.05):
    s = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    return rf"\textbf{{{s}}}" if p < bold_below else s


def fmt_dec(x, digits=4, drop_zero=True):
    s = f"{x:.{digits}f}"
    if drop_zero:
        s = s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "-.", 1)
    return s


def fmt_sci(x):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    return rf"${x / 10**exp:.1f}\!\times\!10^{{{exp}}}$"


def sgn(x, digits=1):
    """Signed number in math mode, so LaTeX typesets a real minus sign."""
    return f"${x:+.{digits}f}$"


def tex_escape(s):
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
            .replace(">", r"$>$").replace("<", r"$<$"))


def write(name, lines):
    path = TAB / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def paired_test(d):
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return 0.0, 1.0
    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(res.statistic), float(res.pvalue)


# ===================================================================== synthetic study
syn_raw = pd.read_csv(OUT / "correlation" / "results_raw.csv")
syn_sum = pd.read_csv(OUT / "correlation" / "summary.csv")
syn_cfg = json.loads((OUT / "correlation" / "config.json").read_text(encoding="utf-8"))
KEY["synthetic_config"] = {k: syn_cfg[k] for k in (
    "rho", "functions", "n_features", "n_train", "n_test", "signal", "n_replicates",
    "n_estimators", "sparsity_penalty", "stability_penalty", "n_bootstraps", "seed")}
KEY["synthetic_versions"] = syn_cfg["versions"]


def cell(df, f, r):
    return df[(df["function"] == f) & np.isclose(df["rho"], r)]


# --- T1: mean test Brier per setting (all eight models + oracle) ---------------------
lines = [r"\begin{tabular}{ll r rrrrrrrr}", r"\toprule",
         r"$f$ & $\rho$ & Oracle & " + " & ".join(SYN_SHORT[m] for m in SYN_MODELS) + r" \\",
         r"\midrule"]
best_counts = {m: 0 for m in SYN_MODELS}
sd_lines = [r"\begin{tabular}{ll rrrrrrrr}", r"\toprule",
            r"$f$ & $\rho$ & " + " & ".join(SYN_SHORT[m] for m in SYN_MODELS) + r" \\",
            r"\midrule"]
blocks = {}  # rows of the mean-Brier table per target function (also used for the slides)
for fi, f in enumerate(FUNCS):
    if fi:
        sd_lines.append(r"\midrule")
    blocks[f] = []
    for ri, r in enumerate(RHOS):
        c = cell(syn_sum, f, r).set_index("model")
        means = {m: c.loc[m, "brier_mean"] for m in SYN_MODELS}
        best = min(means, key=means.get)
        best_counts[best] += 1
        cells = [(rf"\textbf{{{fmt_dec(means[m])}}}" if m == best else fmt_dec(means[m]))
                 for m in SYN_MODELS]
        label = FUNC_TEX[f] if ri == 0 else ""
        oracle = c["oracle_brier_mean"].iloc[0]
        blocks[f].append(f"{label} & {r:g} & {fmt_dec(oracle)} & " + " & ".join(cells) + r" \\")
        sd_lines.append(f"{label} & {r:g} & "
                        + " & ".join(fmt_dec(c.loc[m, 'brier_std']) for m in SYN_MODELS) + r" \\")


def brier_table(funcs):
    out = list(lines)
    for k, f in enumerate(funcs):
        if k:
            out.append(r"\midrule")
        out += blocks[f]
    return out + [r"\bottomrule", r"\end{tabular}"]


write("synthetic_brier_a.tex", brier_table(FUNCS[:2]))  # slides: two halves of the table
write("synthetic_brier_b.tex", brier_table(FUNCS[2:]))
lines = brier_table(FUNCS)
sd_lines += [r"\bottomrule", r"\end{tabular}"]
write("synthetic_brier.tex", lines)
write("synthetic_brier_sd.tex", sd_lines)
KEY["synthetic_best_model_counts"] = best_counts

# --- T2: primary paired tests (saved by exp_statistical_analysis.py) + global Holm ----
frames = []
for f in FUNCS:
    for r in RHOS:
        t = pd.read_csv(OUT / "statistical" / f / f"rho_{r}" / "paired_wilcoxon.csv")
        t.insert(0, "rho", r)
        t.insert(0, "function", f)
        frames.append(t)
prim = pd.concat(frames, ignore_index=True)
prim["p_holm_global"] = holm(prim["p_raw"])
prim.to_csv(TAB / "synthetic_primary_tests.csv", index=False)

comparators = [L0, UNIF, SHUF]
lines = [r"\begin{tabular}{ll rrr rrr rrr}", r"\toprule",
         r" & & \multicolumn{3}{c}{vs.\ L0} & \multicolumn{3}{c}{vs.\ uniform control}"
         r" & \multicolumn{3}{c}{vs.\ shuffled control} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"$f$ & $\rho$ & $\bar\Delta\!\times\!10^{4}$ & wins & $p_{\mathrm{Holm}}$"
         r" & $\bar\Delta\!\times\!10^{4}$ & wins & $p_{\mathrm{Holm}}$"
         r" & $\bar\Delta\!\times\!10^{4}$ & wins & $p_{\mathrm{Holm}}$ \\",
         r"\midrule"]
for fi, f in enumerate(FUNCS):
    if fi:
        lines.append(r"\midrule")
    for ri, r in enumerate(RHOS):
        row = [FUNC_TEX[f] if ri == 0 else "", f"{r:g}"]
        for comp in comparators:
            t = prim[(prim["function"] == f) & np.isclose(prim["rho"], r)
                     & (prim["competitor"] == comp)].iloc[0]
            row += [sgn(1e4 * t["mean_delta_brier"]), f"{int(t['full_better_count'])}",
                    fmt_p(t["p_holm_global"])]
        lines.append(" & ".join(row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("synthetic_primary_tests.tex", lines)

summary_prim = {}
for comp in comparators:
    t = prim[prim["competitor"] == comp]
    summary_prim[comp] = dict(
        n_cells=int(len(t)),
        mean_delta_min=float(t["mean_delta_brier"].min()),
        mean_delta_max=float(t["mean_delta_brier"].max()),
        mean_delta_avg=float(t["mean_delta_brier"].mean()),
        sig_global_full_better=int(((t["p_holm_global"] < 0.05) & (t["mean_delta_brier"] < 0)).sum()),
        sig_global_full_worse=int(((t["p_holm_global"] < 0.05) & (t["mean_delta_brier"] > 0)).sum()),
        sig_within_full_better=int(((t["p_holm"] < 0.05) & (t["mean_delta_brier"] < 0)).sum()),
        sig_within_full_worse=int(((t["p_holm"] < 0.05) & (t["mean_delta_brier"] > 0)).sum()),
        cells_full_better_on_average=int((t["mean_delta_brier"] < 0).sum()),
        total_wins=int(t["full_better_count"].sum()),
        total_pairs=int(t["n_pairs"].sum()),
    )
KEY["synthetic_primary_tests"] = summary_prim

# size of the control differences relative to the between-replicate sd of the full model
sd_full = syn_sum[syn_sum["model"] == FULL].set_index(["function", "rho"])["brier_std"]
ctrl = prim[prim["competitor"].isin([UNIF, SHUF])].copy()
ctrl["ratio"] = [abs(row.mean_delta_brier) / sd_full.loc[(row.function, row.rho)]
                 for row in ctrl.itertuples()]
KEY["synthetic_control_delta_over_sd"] = dict(
    mean=float(ctrl["ratio"].mean()), median=float(ctrl["ratio"].median()),
    max=float(ctrl["ratio"].max()))

# --- variability: between-replicate sd of the test Brier loss -------------------------
sd = syn_sum.pivot_table(index=["function", "rho"], columns="model", values="brier_std")
oblique_cols = [OBL, L0, STAB, FULL, UNIF, SHUF, TRAIN]
KEY["synthetic_sd"] = dict(
    full_gt_L0_settings=int((sd[FULL] > sd[L0]).sum()),
    full_gt_L0_rho0=int((sd.xs(0.0, level="rho")[FULL] > sd.xs(0.0, level="rho")[L0]).sum()),
    rho0={f: {SYN_SHORT[m]: float(sd.loc[(f, 0.0), m]) for m in (L0, FULL, STAB, TRAIN, OBL)}
          for f in FUNCS},
    rho09={f: {SYN_SHORT[m]: float(sd.loc[(f, 0.9), m]) for m in (L0, FULL)} for f in FUNCS},
    lowest_sd_counts={SYN_SHORT[m]: int(c) for m, c in
                      sd[oblique_cols].idxmin(axis=1).value_counts().items()},
    mean_sd={SYN_SHORT[m]: float(sd[m].mean()) for m in SYN_MODELS},
)

# --- T3: exploratory paired contrasts computed from the raw results ------------------
wide = syn_raw.pivot_table(index=["function", "rho", "replicate"], columns="model",
                           values="brier").reset_index()
contrasts = [
    (STAB, L0, "Stability only $-$ L0 only"),
    (STAB, OBL, "Stability only $-$ unpenalized"),
    (L0, OBL, "L0 only $-$ unpenalized"),
    (FULL, OBL, "L0+stability $-$ unpenalized"),
    (TRAIN, OBL, "Training selection $-$ unpenalized"),
    (OBL, RF, "Unpenalized oblique $-$ RF"),
]
rows = []
for a, b, label in contrasts:
    for f in FUNCS:
        for r in RHOS:
            w = wide[(wide["function"] == f) & np.isclose(wide["rho"], r)]
            d = (w[a] - w[b]).to_numpy()
            stat, p = paired_test(d)
            rows.append(dict(contrast=label, a=a, b=b, function=f, rho=r,
                             mean_delta=float(d.mean()), wins=int((d < 0).sum()), p=p))
expl = pd.DataFrame(rows)
expl["p_holm"] = holm(expl["p"])
expl.to_csv(TAB / "synthetic_exploratory_tests.csv", index=False)
lines = [r"\begin{tabular}{l rrr rrr}", r"\toprule",
         r"Contrast ($A-B$) & \multicolumn{3}{c}{$\bar\Delta\!\times\!10^{4}$ over 20 settings}"
         r" & \multicolumn{3}{c}{settings (Holm, 120 tests)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r" & min & median & max & $A$ better & $A$ worse & n.s. \\", r"\midrule"]
expl_key = {}
for _, _, label in contrasts:
    t = expl[expl["contrast"] == label]
    better = int(((t["p_holm"] < 0.05) & (t["mean_delta"] < 0)).sum())
    worse = int(((t["p_holm"] < 0.05) & (t["mean_delta"] > 0)).sum())
    lines.append(f"{label} & {sgn(1e4 * t['mean_delta'].min())} & "
                 f"{sgn(1e4 * t['mean_delta'].median())} & {sgn(1e4 * t['mean_delta'].max())} & "
                 f"{better} & {worse} & {len(t) - better - worse}" + r" \\")
    expl_key[label.replace("$-$", "-")] = dict(
        better=better, worse=worse, ns=len(t) - better - worse,
        min=float(t["mean_delta"].min()), median=float(t["mean_delta"].median()),
        max=float(t["mean_delta"].max()))
lines += [r"\bottomrule", r"\end{tabular}"]
write("synthetic_exploratory_tests.tex", lines)
KEY["synthetic_exploratory_tests"] = expl_key

# --- T4: structure, selection and cost per model (averaged over the 20 settings) -----
syn_sum["brier_rank"] = syn_sum.groupby(["function", "rho"])["brier_mean"].rank()
struct = syn_sum.groupby("model").agg(
    rank=("brier_rank", "mean"), nodes=("nodes_per_tree_mean", "mean"),
    support=("mean_support_size_mean", "mean"), relevant=("relevant_fraction_mean", "mean"),
    root_relevant=("root_relevant_fraction_mean", "mean"),
    accuracy=("accuracy_mean", "mean"), ap=("average_precision_mean", "mean"),
    fit=("fit_seconds_mean", "mean")).reindex(SYN_MODELS)
lines = [r"\begin{tabular}{l rrrrrrr}", r"\toprule",
         r"Model & Brier rank & Nodes/tree & $\|w\|_0$ & Rel.\ share & "
         r"Rel.\ (root) & Accuracy & Fit [s] \\", r"\midrule"]
for m in SYN_MODELS:
    s = struct.loc[m]
    na = "--"
    lines.append(
        f"{SYN_LONG[m]} & {s['rank']:.2f} & {s['nodes']:.1f} & "
        f"{na if np.isnan(s['support']) else f'{s.support:.2f}'} & "
        f"{na if np.isnan(s['relevant']) else f'{s.relevant:.3f}'} & "
        f"{na if np.isnan(s['root_relevant']) else f'{s.root_relevant:.3f}'} & "
        f"{s['accuracy']:.3f} & {s['fit']:.2f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("synthetic_structure.tex", lines)
KEY["synthetic_structure"] = struct.round(4).to_dict(orient="index")
KEY["synthetic_fit_time_ratio_full_over_L0"] = float(struct.loc[FULL, "fit"] / struct.loc[L0, "fit"])

# relevant-feature share per function (averaged over rho), with the chance level
p = syn_cfg["n_features"]
oblique_models = [OBL, L0, STAB, FULL, TRAIN]
lines = [r"\begin{tabular}{l r " + "r" * len(oblique_models) + "}", r"\toprule",
         r"$f$ & chance & " + " & ".join(SYN_SHORT[m] for m in oblique_models) + r" \\",
         r"\midrule"]
for f in FUNCS:
    c = syn_sum[syn_sum["function"] == f].groupby("model")["relevant_fraction_mean"].mean()
    lines.append(f"{FUNC_TEX[f]} & {RELEVANT[f] / p:.3f} & "
                 + " & ".join(f"{c[m]:.3f}" for m in oblique_models) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("synthetic_relevant_share.tex", lines)

# ======================================================== regularization paths
lines = [r"\begin{tabular}{ll rr rr r}", r"\toprule",
         r"$f$ & Penalty family & best $\lambda$ & best $\gamma$ & Brier ($\pm$ s.e.) & "
         r"Nodes/tree & Mean fit [s] \\", r"\midrule"]
reg_key = {}
for fi, f in enumerate(["axis", "oblique2"]):
    reg = pd.read_csv(OUT / "regularization" / f / "summary.csv")
    if fi:
        lines.append(r"\midrule")
    ref = reg[(reg["model"] == "L0") & (reg["sparsity_penalty"] == 0)].iloc[0]
    lines.append(f"{FUNC_TEX[f]} & unpenalized & 0 & 0 & {fmt_dec(ref.brier_mean, 5)} "
                 f"($\\pm${fmt_dec(ref.brier_se, 5)}) & {ref.nodes_per_tree_mean:.1f} & "
                 f"{ref.fit_seconds_mean:.2f}" + r" \\")
    reg_key[f] = {"unpenalized_brier": float(ref.brier_mean)}
    for fam, label in [("L0", "L0 only"), ("Stability", "feature stability"),
                       ("Uniform", "uniform control"), ("Shuffled", "shuffled control")]:
        g = reg[reg["model"] == fam]
        b = g.loc[g["brier_mean"].idxmin()]
        lines.append(f" & {label} & {b.sparsity_penalty:g} & {b.stability_penalty:g} & "
                     f"{fmt_dec(b.brier_mean, 5)} ($\\pm${fmt_dec(b.brier_se, 5)}) & "
                     f"{b.nodes_per_tree_mean:.1f} & {g.fit_seconds_mean.mean():.2f}" + r" \\")
        reg_key[f][fam] = dict(best_lambda=float(b.sparsity_penalty),
                               best_gamma=float(b.stability_penalty),
                               best_brier=float(b.brier_mean), best_se=float(b.brier_se),
                               nodes=float(b.nodes_per_tree_mean),
                               mean_fit=float(g.fit_seconds_mean.mean()),
                               brier_range=[float(g.brier_mean.min()), float(g.brier_mean.max())])
    reg_key[f]["fit_ratio_stability_over_L0"] = (
        reg_key[f]["Stability"]["mean_fit"] / reg_key[f]["L0"]["mean_fit"])
lines += [r"\bottomrule", r"\end{tabular}"]
write("regularization_best.tex", lines)
KEY["regularization"] = reg_key

# ===================================================================== OpenML
om = pd.read_csv(OUT / "openml" / "tables" / "openml_benchmark_summary.csv")
# keep "None" (unlimited max_depth) as a string instead of pandas' default NaN
om_res = pd.read_csv(OUT / "openml" / "tables" / "openml_benchmark_results.csv",
                     keep_default_na=False, na_values=[""])
om_macro = pd.read_csv(OUT / "openml" / "tables" / "openml_benchmark_macro.csv").set_index("model")
om_trials = pd.read_csv(OUT / "openml" / "tables" / "openml_search_trials.csv")
datasets = list(dict.fromkeys(om["dataset"]))

# --- datasets -------------------------------------------------------------------------
lines = [r"\begin{tabular}{l r rrr l r}", r"\toprule",
         r"Dataset & OpenML id & $n$ (total) & $n$ (used) & $p$ (raw) & Positive class & "
         r"Pos.\ rate \\", r"\midrule"]
for d in datasets:
    s = om[om["dataset"] == d].iloc[0]
    lines.append(f"{tex_escape(d)} & {s.openml_id} & {s.n_samples_total} & {s.n_samples} & "
                 f"{s.n_features} & {tex_escape(s.positive_class)} & {s.positive_rate:.2f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("openml_datasets.tex", lines)


def om_metric_table(metric, lower_better, digits, name):
    lines = [r"\begin{tabular}{l rrrrr}", r"\toprule",
             r"Dataset & " + " & ".join(OML_SHORT[m] for m in OML_MODELS) + r" \\", r"\midrule"]
    for d in datasets:
        s = om[om["dataset"] == d].set_index("model")
        vals = {m: s.loc[m, f"{metric}_mean"] for m in OML_MODELS}
        best = (min if lower_better else max)(vals, key=vals.get)
        cells = []
        for m in OML_MODELS:
            txt = f"{fmt_dec(vals[m], digits)} ({fmt_dec(s.loc[m, f'{metric}_std'], digits)})"
            cells.append(rf"\textbf{{{txt}}}" if m == best else txt)
        lines.append(f"{tex_escape(d)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Macro mean & " + " & ".join(
        fmt_dec(om_macro.loc[m, f"{metric}_macro_mean"], digits) for m in OML_MODELS) + r" \\")
    lines.append(r"Average rank & " + " & ".join(
        f"{om_macro.loc[m, f'{metric}_avg_rank']:.2f}" for m in OML_MODELS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, lines)


om_metric_table("brier", True, 3, "openml_brier.tex")
om_metric_table("accuracy", False, 3, "openml_accuracy.tex")
om_metric_table("f1", False, 3, "openml_f1.tex")
om_metric_table("average_precision", False, 3, "openml_ap.tex")
KEY["openml_macro"] = om_macro.round(4).to_dict(orient="index")

# --- dataset-level paired tests ---------------------------------------------------------
piv = {metric: om.pivot(index="dataset", columns="model", values=f"{metric}_mean").loc[datasets]
       for metric in ("brier", "accuracy", "f1", "average_precision")}
fried = friedmanchisquare(*[piv["brier"][m].to_numpy() for m in OML_MODELS])
KEY["openml_friedman_brier"] = dict(statistic=float(fried.statistic), p=float(fried.pvalue))

pairs = [(O_FULL, O_L0, True), (O_FULL, O_BASE, False), (O_L0, O_BASE, False),
         (O_FULL, O_RF, False), (O_L0, O_RF, False), (O_BASE, O_RFM, False)]
rows = []
for a, b, primary in pairs:
    d = (piv["brier"][a] - piv["brier"][b]).to_numpy()
    stat, pval = paired_test(d)
    rel = 100 * (piv["brier"][a] / piv["brier"][b] - 1).to_numpy()
    rows.append(dict(a=a, b=b, primary=primary, mean=d.mean(), median=float(np.median(d)),
                     rel_median=float(np.median(rel)), wins=int((d < 0).sum()),
                     ties=int((d == 0).sum()), losses=int((d > 0).sum()), stat=stat, p=pval))
tests = pd.DataFrame(rows)
expl_mask = ~tests["primary"]
tests.loc[expl_mask, "p_holm"] = holm(tests.loc[expl_mask, "p"])
tests.loc[~expl_mask, "p_holm"] = tests.loc[~expl_mask, "p"]
tests.to_csv(TAB / "openml_tests.csv", index=False)
lines = [r"\begin{tabular}{l rrr r rr}", r"\toprule",
         r"Comparison ($A-B$) & $\bar\Delta\!\times\!10^{3}$ & med.\ $\Delta\!\times\!10^{3}$ & "
         r"med.\ rel.\ [\%] & W/T/L & $p$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
for _, t in tests.iterrows():
    label = f"{OML_SHORT[t.a]} $-$ {OML_SHORT[t.b]}"  # first row = primary test
    lines.append(f"{label} & {sgn(1e3 * t['mean'], 2)} & {sgn(1e3 * t['median'], 2)} & "
                 f"{sgn(t.rel_median)} & {t.wins}/{t.ties}/{t.losses} & {fmt_p(t.p)} & "
                 f"{'--' if t.primary else fmt_p(t.p_holm)}" + r" \\")
    if t.primary:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
write("openml_tests.tex", lines)
KEY["openml_tests_brier"] = tests.round(6).to_dict(orient="records")

# primary pair on secondary metrics (higher is better, so a win is d > 0)
sec = {}
for metric in ("accuracy", "f1", "average_precision"):
    d = (piv[metric][O_FULL] - piv[metric][O_L0]).to_numpy()
    stat, pval = paired_test(d)
    sec[metric] = dict(mean=float(d.mean()), wins=int((d > 0).sum()), ties=int((d == 0).sum()),
                       losses=int((d < 0).sum()), p=pval)
KEY["openml_full_vs_L0_secondary"] = sec

# --- tuning outcomes and cost -----------------------------------------------------------
lines = [r"\begin{tabular}{l rr rrl rr}", r"\toprule",
         r" & \multicolumn{2}{c}{L0 tuned} & \multicolumn{3}{c}{L0+stability tuned} & "
         r"\multicolumn{2}{c}{cost ratio (stab./L0)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
         r"Dataset & $\tilde\lambda$ & $\lambda{=}0$ & $\tilde\gamma$ & $\gamma{=}0$ & "
         r"max\_depth & fit & search \\", r"\midrule"]
ratios_fit, ratios_search = [], []
for d in datasets:
    s = om[om["dataset"] == d].set_index("model")
    l0, fu = s.loc[O_L0], s.loc[O_FULL]
    rf_, rs_ = fu.fit_seconds_mean / l0.fit_seconds_mean, fu.search_seconds_mean / l0.search_seconds_mean
    ratios_fit.append(rf_)
    ratios_search.append(rs_)
    lines.append(f"{tex_escape(d)} & {fmt_sci(float(l0.tuned_lambda_median))} & "
                 f"{int(l0.tuned_lambda_zero_folds)}/5 & {fmt_sci(float(fu.tuned_gamma_median))} & "
                 f"{int(fu.tuned_gamma_zero_folds)}/5 & {tex_escape(fu.tuned_max_depth_counts)} & "
                 f"{rf_:.1f} & {rs_:.1f}" + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("openml_tuning.tex", lines)

full_rows = om_res[om_res["model"] == O_FULL]
gam = full_rows["stability_penalty"].astype(float)
KEY["openml_tuning"] = dict(
    folds=int(len(gam)), gamma_zero=int((gam == 0).sum()),
    gamma_lt_1e3=int(((gam > 0) & (gam < 1e-3)).sum()),
    gamma_1e3_1e2=int(((gam >= 1e-3) & (gam < 1e-2)).sum()),
    gamma_ge_1e2=int((gam >= 1e-2).sum()),
    gamma_median_all=float(gam.median()),
    fit_ratio_median=float(np.median(ratios_fit)), fit_ratio_range=[float(min(ratios_fit)), float(max(ratios_fit))],
    search_ratio_median=float(np.median(ratios_search)),
    search_ratio_range=[float(min(ratios_search)), float(max(ratios_search))],
    depth_counts_full=full_rows["max_depth"].astype(str).value_counts().to_dict(),
    depth_counts_L0=om_res[om_res["model"] == O_L0]["max_depth"].astype(str).value_counts().to_dict(),
    leaf_counts_full=full_rows["min_samples_leaf"].astype(float).astype(int).astype(str)
    .value_counts().to_dict(),
    leaf_counts_L0=om_res[om_res["model"] == O_L0]["min_samples_leaf"].astype(float)
    .astype(int).astype(str).value_counts().to_dict(),
)

# fold-level variability and per-dataset winners
sd_om = om.pivot(index="dataset", columns="model", values="brier_std").loc[datasets]
best_model = piv["brier"].idxmin(axis=1)
KEY["openml_variability"] = dict(
    full_sd_lt_L0=int((sd_om[O_FULL] < sd_om[O_L0]).sum()),
    full_sd_gt_L0=int((sd_om[O_FULL] > sd_om[O_L0]).sum()),
    mean_sd={OML_SHORT[m]: float(sd_om[m].mean()) for m in OML_MODELS},
)
KEY["openml_best_model_per_dataset"] = {d: OML_SHORT[m] for d, m in best_model.items()}
KEY["openml_rf_better_than_L0_datasets"] = [
    d for d in datasets if piv["brier"].loc[d, O_RF] < piv["brier"].loc[d, O_L0]]
KEY["openml_rf_worse_than_L0_datasets"] = [
    d for d in datasets if piv["brier"].loc[d, O_RF] > piv["brier"].loc[d, O_L0]]

# inner-CV sensitivity: excess inner Brier over the best trial in the same outer fold
tr = om_trials[om_trials["model"] == O_FULL].copy()
tr["excess"] = tr["brier"] - tr.groupby(["dataset", "fold"])["brier"].transform("min")
bins = [(-1, 0, r"$\gamma=0$"), (0, 1e-3, r"$0<\gamma<10^{-3}$"),
        (1e-3, 1e-2, r"$10^{-3}\le\gamma<10^{-2}$"), (1e-2, 1, r"$\gamma\ge10^{-2}$")]
lines = [r"\begin{tabular}{l rrr}", r"\toprule",
         r"Sampled $\gamma$ & trials & median excess inner Brier $\times10^{3}$ & "
         r"selected configs \\", r"\midrule"]
sens = {}
for lo, hi, label in bins:
    if hi == 0:
        mask = tr["stability_penalty"] == 0
    else:
        mask = (tr["stability_penalty"] > max(lo, 0)) & (tr["stability_penalty"] < hi) if lo == 0 \
            else (tr["stability_penalty"] >= lo) & (tr["stability_penalty"] < hi)
    g = tr[mask]
    lines.append(f"{label} & {len(g)} & {1e3 * g['excess'].median():.2f} & "
                 f"{int(g['selected'].sum())}" + r" \\")
    sens[label] = dict(trials=int(len(g)), median_excess=float(g["excess"].median()),
                       selected=int(g["selected"].sum()))
lines += [r"\bottomrule", r"\end{tabular}"]
write("openml_gamma_sensitivity.tex", lines)
KEY["openml_gamma_sensitivity"] = sens

# structure of fitted forests (means over datasets)
struct_om = om.groupby("model")[["nodes_per_tree_mean", "mean_support_size_mean",
                                 "fit_seconds_mean", "search_seconds_mean"]].mean()
KEY["openml_structure_mean_over_datasets"] = struct_om.round(3).to_dict(orient="index")

(TAB / "key_numbers.json").write_text(json.dumps(KEY, indent=2, default=float), encoding="utf-8")
print(f"wrote {(TAB / 'key_numbers.json').relative_to(ROOT)}")
