"""Visual, beginner-friendly demonstration of the oblique-forest estimators.

This is a usage example, not a scientific benchmark. The models are fitted on
one synthetic dataset and evaluated on one held-out test split. The outputs
illustrate model behavior; they are not evidence of superiority.

Run from the repository root:

    uv run --with matplotlib python examples/visual_demo.py
    uv run --with matplotlib python examples/visual_demo.py --dataset moons
    uv run --with matplotlib python examples/visual_demo.py --show

Plots are saved to ``outputs/visual_demo/`` by default. Change with --output-dir.
Matplotlib is an optional dependency needed only for this visual example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.special import expit
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from stable_oblique_forests import ObliqueRandomForestClassifier


# These are small settings for a quick demonstration, not tuned values.
SEED = 42


def create_dataset(name: str, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two numerical features and binary labels.

    diagonal: probability changes smoothly across a 45-degree linear boundary.
    moons: a nonlinear classification problem from scikit-learn.
    """
    if name == "diagonal":
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, 2))
        true_p = expit(3.5 * (X[:, 0] + X[:, 1]) / np.sqrt(2))
        y = rng.binomial(1, true_p).astype(int)
        return X, y
    if name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.23, random_state=seed)
        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    if name == "mixed":
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2.5, 2.5, size=(n_samples, 2))

        margin = np.where(
            X[:, 0] < -0.25,
            X[:, 1] - 0.25,
            (X[:, 0] + X[:, 1] - 0.4) / np.sqrt(2),
        )

        true_p = expit(5.0 * margin)
        y = rng.binomial(1, true_p).astype(int)
        return X, y
    raise ValueError(f"Unknown dataset: {name!r}")


def create_models(
    seed: int,
    n_estimators: int,
    n_bootstraps: int,
    sparsity_penalty: float,
    stability_penalty: float,
) -> dict[str, Pipeline]:
    """Create the classifiers used in the visual comparison."""

    common_oblique_params = dict(
        n_estimators=n_estimators,
        support_sizes=(1, 2),
        n_directions_per_k=8,
        max_features=None,
        min_samples_split=40,
        min_samples_leaf=10,
        min_assessment_samples=10,
        assessment_fraction=0.20,
        max_depth=5,
        random_state=seed,
        n_jobs=1,
    )

    estimators = {
        "Decision tree (CART)": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            random_state=seed,
        ),

        "Random forest (sklearn)": RandomForestClassifier(
            n_estimators=max(n_estimators, 30),
            max_depth=5,
            min_samples_leaf=10,
            random_state=seed,
            n_jobs=1,
        ),

        "Oblique forest (no penalty)": ObliqueRandomForestClassifier(
            **common_oblique_params,
            sparsity_penalty=0.0,
            stability_penalty=0.0,
        ),

        "Sparse oblique forest (L0)": ObliqueRandomForestClassifier(
            **common_oblique_params,
            sparsity_penalty=sparsity_penalty,
            stability_penalty=0.0,
        ),

        "Stability-aware oblique forest": ObliqueRandomForestClassifier(
            **common_oblique_params,
            sparsity_penalty=sparsity_penalty,
            stability_penalty=stability_penalty,
            n_bootstraps=n_bootstraps,
            stability_mode="feature",
        ),
    }

    return {
        name: Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ])
        for name, estimator in estimators.items()
    }


def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> list[dict]:
    results = []
    for name, model in models.items():
        print(f"Fitting {name} ...", flush=True)
        model.fit(X_train, y_train)
        p1 = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        results.append(
            {
                "name": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "brier": brier_score_loss(y_test, p1),
            }
        )
    return results


def print_results(results: list[dict]) -> None:
    print("\nHeld-out test results (single split, for demonstration only)")
    print(f"{'Model':37} {'Accuracy':>9} {'Bal. acc.':>10} {'Brier':>9}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:37} {r['accuracy']:9.3f} "
            f"{r['balanced_accuracy']:10.3f} {r['brier']:9.3f}"
        )
    print("Brier score: lower is better. Accuracy: higher is better.\n")


def plot_probability_maps(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    output_path: Path,
    grid_size: int,
    dataset: str,
) -> None:
    """Plot model probabilities and the true data-generating probability."""

    all_X = np.vstack([X_train, X_test])
    mins = all_X.min(axis=0) - 0.4
    maxs = all_X.max(axis=0) + 0.4

    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], grid_size),
        np.linspace(mins[1], maxs[1], grid_size),
    )
    mesh = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.ravel()
    levels = np.linspace(0, 1, 21)
    contour = None

    for ax, (name, pipeline) in zip(axes, models.items()):
        proba = pipeline.predict_proba(mesh)[:, 1].reshape(xx.shape)

        contour = ax.contourf(
            xx, yy, proba,
            levels=levels,
            cmap="coolwarm",
            norm=Normalize(vmin=0, vmax=1),
        )

        ax.contour(
            xx, yy, proba,
            levels=[0.5],
            colors="black",
            linewidths=1.1,
        )

        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=10,
            alpha=0.45,
            edgecolors="none",
        )

        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Feature $x_1$")
        ax.set_ylabel("Feature $x_2$")

    # The sixth panel shows the true probability, when known.
    ax = axes[-1]

    if dataset == "mixed":
        margin = np.where(
            mesh[:, 0] < -0.25,
            mesh[:, 1] - 0.25,
            (mesh[:, 0] + mesh[:, 1] - 0.4) / np.sqrt(2),
        )
        true_proba = expit(5.0 * margin).reshape(xx.shape)

    elif dataset == "diagonal":
        margin = (mesh[:, 0] + mesh[:, 1]) / np.sqrt(2)
        true_proba = expit(3.5 * margin).reshape(xx.shape)

    else:
        true_proba = None

    if true_proba is not None:
        ax.contourf(
            xx, yy, true_proba,
            levels=levels,
            cmap="coolwarm",
            norm=Normalize(vmin=0, vmax=1),
        )

        ax.contour(
            xx, yy, true_proba,
            levels=[0.5],
            colors="black",
            linewidths=1.1,
        )

        ax.set_title("True probability", fontsize=11)
        ax.set_xlabel("Feature $x_1$")
        ax.set_ylabel("Feature $x_2$")
    else:
        ax.axis("off")

    fig.colorbar(
        contour,
        ax=axes.tolist(),
        shrink=0.75,
        pad=0.02,
        label="P(y = 1)",
    )

    fig.suptitle(
        f"Decision probability maps: {dataset}",
        fontsize=15,
    )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metrics(results: list[dict], output_path: Path) -> None:
    """Compare accuracy and Brier score on the held-out test set."""

    short_names = [
        "CART",
        "sklearn RF",
        "Oblique",
        "Oblique + L0",
        "Oblique + L0 + stability",
    ]

    x = np.arange(len(results))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        x,
        [r["accuracy"] for r in results],
    )
    axes[0].set_title("Test accuracy (higher is better)")
    axes[0].set_ylim(0, 1)

    axes[1].bar(
        x,
        [r["brier"] for r in results],
    )
    axes[1].set_title("Test Brier score (lower is better)")
    axes[1].set_ylim(bottom=0)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Illustrative comparison on one train/test split")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_root_stability(stable_pipeline: Pipeline, output_path: Path) -> bool:
    """Average local bootstrap selection frequencies at the roots of split trees.

    This uses the research estimator's node_diagnostics_ attribute. It is NOT
    global feature importance and is not normalized to sum to one.
    """
    forest = stable_pipeline.named_steps["classifier"]
    root_frequencies = []
    for tree in forest.estimators_:
        root = next(
            (d for d in tree.node_diagnostics_ if d["node_id"] == 0), None
        )
        if root is not None and root["n_valid_bootstraps"] > 0:
            root_frequencies.append(
                [root["feature_frequencies"].get(j, 0.0) for j in range(forest.n_features_in_)]
            )
    if not root_frequencies:
        return False

    avg = np.asarray(root_frequencies).mean(axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(np.arange(len(avg)), avg)
    ax.set_xticks(np.arange(len(avg)), [f"$x_{{{i + 1}}}$" for i in range(len(avg))])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean bootstrap selection frequency")
    ax.set_title(f"Stability at root nodes ({len(root_frequencies)} trees with splits)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def print_model_details(stable_pipeline: Pipeline, X_test: np.ndarray) -> None:
    forest = stable_pipeline.named_steps["classifier"]
    print("Our forest: an overview of fitted attributes")
    print("  Number of trees:        ", len(forest.estimators_))
    print("  Total number of nodes:  ", forest.node_count_)
    print("  Node-level bootstrap fits:", forest.n_bootstrap_fits_)
    print("  Split counts per feature:", forest.feature_usage_.tolist())
    print("  Heuristic importances:  ", np.round(forest.feature_importances_, 3))
    transformed_test = stable_pipeline.named_steps["scaler"].transform(X_test[:5])
    print("  Leaf ID matrix shape:   ", forest.apply(transformed_test).shape)
    print("  Predicted probabilities:\n", np.round(stable_pipeline.predict_proba(X_test[:5]), 3))
    print("  Fitted parameter gamma: ", forest.get_params()["stability_penalty"])
    print("  sklearn.clone works:    ", isinstance(clone(forest), ObliqueRandomForestClassifier))
    print("  feature_usage_ counts splits; it does not measure causal importance.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("mixed", "diagonal", "moons"),
        default="mixed",
        help="Synthetic 2D dataset.",
    )
    parser.add_argument("--n-samples", type=int, default=650)
    parser.add_argument("--n-estimators", type=int, default=16)
    parser.add_argument("--n-bootstraps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grid-size", type=int, default=110)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/visual_demo"),
        help="Where to save PNG files (relative to current directory by default).",
    )
    parser.add_argument("--show", action="store_true", help="Open saved plots in a GUI.")

    parser.add_argument(
        "--sparsity-penalty", type=float, default=0.003
    )
    parser.add_argument(
        "--stability-penalty", type=float, default=0.007
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples < 100 or args.n_estimators < 1 or args.n_bootstraps < 1:
        raise ValueError("Require n_samples >= 100, n_estimators >= 1, n_bootstraps >= 1.")
    if args.grid_size < 30:
        raise ValueError("--grid-size must be at least 30.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = create_dataset(args.dataset, args.n_samples, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=args.seed
    )
    print(f"Dataset: {args.dataset}, train={len(y_train)}, test={len(y_test)}")
    models = create_models(
        seed=args.seed,
        n_estimators=args.n_estimators,
        n_bootstraps=args.n_bootstraps,
        sparsity_penalty=args.sparsity_penalty,
        stability_penalty=args.stability_penalty,
    )
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    print_results(results)

    stable_pipeline = models["Stability-aware oblique forest"]
    print_model_details(stable_pipeline, X_test)

    plots = {
        "Decision maps": args.output_dir / f"{args.dataset}_decision_maps.png",
        "Test metrics": args.output_dir / f"{args.dataset}_metrics.png",
        "Root stability": args.output_dir / f"{args.dataset}_root_stability.png",
    }
    print("Generating plots...")
    plot_probability_maps(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        output_path=plots["Decision maps"],
        grid_size=args.grid_size,
        dataset=args.dataset,
    )
    plot_metrics(results, plots["Test metrics"])
    has_stability_plot = plot_root_stability(stable_pipeline, plots["Root stability"])
    if not has_stability_plot:
        plots.pop("Root stability")
        print("No eligible root split: skipping the root stability plot.")

    for name, path in plots.items():
        print(f"  {name}: {path.resolve()}")

    if args.show:
        # Loading the already-saved PNGs keeps plotting behavior consistent.
        for path in plots.values():
            image = plt.imread(path)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(path.name)
        plt.show()


if __name__ == "__main__":
    main()
