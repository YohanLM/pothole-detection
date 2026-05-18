"""Plot detection metrics as a function of training set size.

For each experiment directory, reads results.json for mAP metrics and re-runs
validation on the saved best.pt weights to extract optimal-F1 precision/recall/F1.
Results are cached back into results.json to avoid re-running val() on subsequent calls.

Usage:
    python plot_results.py [experiment_dir ...]

    # single experiment (default: sq_c30_m15_col)
    python plot_results.py

    # compare multiple experiments
    python plot_results.py experiments/sq_c30_m15_col experiments/sq_c30_m15_grey
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

DEVICE = "mps"

METRICS = [
    ("mAP@.5",        "mAP @ IoU=0.50",                 "mAP"),
    ("mAP@.5:.95",    "mAP @ IoU=0.50:0.95",            "mAP"),
    ("inference_ms",  "Inference time (ms / image)",     "ms"),
    ("f1_optimal",    "F1 at optimal confidence",        "F1"),
    ("precision_opt", "Precision at optimal confidence", "Precision"),
    ("recall_opt",    "Recall at optimal confidence",    "Recall"),
]


def enrich_with_f1(run_dir: Path, results: list[dict]) -> list[dict]:
    """Re-run val() on saved best.pt for any entry missing F1/P/R metrics."""
    from ultralytics import YOLO

    detect_dir = run_dir / "detect"
    splits_dir = run_dir / "splits"
    changed = False

    for r in results:
        if "f1_optimal" in r:
            continue
        size = r["size"]
        weight = detect_dir / f"size_{size}" / "weights" / "best.pt"
        yaml_path = splits_dir / f"dataset_{size}.yaml"

        if not weight.exists():
            print(f"  [skip size={size}] no weights at {weight}")
            continue
        if not yaml_path.exists():
            print(f"  [skip size={size}] no dataset yaml at {yaml_path}")
            continue

        print(f"  Running val for size={size}…")
        model = YOLO(str(weight))
        metrics = model.val(
            data=str(yaml_path),
            device=DEVICE,
            verbose=False,
            plots=False,
        )
        box = metrics.box
        # box.f1 is shape (nc,) at the optimal-F1 confidence threshold
        f1_arr = np.asarray(box.f1)
        p_arr  = np.asarray(box.p)
        r_arr  = np.asarray(box.r)

        r["f1_optimal"]    = float(f1_arr.mean())
        r["precision_opt"] = float(p_arr.mean())
        r["recall_opt"]    = float(r_arr.mean())
        changed = True

    return results, changed


def load_results(run_dir: Path) -> list[dict]:
    results_file = run_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No results.json in {run_dir}")
    results = json.loads(results_file.read_text())
    return results


def save_results(run_dir: Path, results: list[dict]) -> None:
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))


def plot_experiments(experiments: dict[str, list[dict]]) -> None:
    n_metrics = len(METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax_idx, (key, title, ylabel) in enumerate(METRICS):
        ax = axes[ax_idx]
        for exp_idx, (label, results) in enumerate(experiments.items()):
            xs = [r["size"] for r in results if key in r]
            ys = [r[key]   for r in results if key in r]
            if not xs:
                continue
            color = colors[exp_idx % len(colors)]
            ax.plot(xs, ys, "o-", color=color, label=label, linewidth=2, markersize=6)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training set size")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        # set ticks at the actual x values across all experiments
        all_xs = sorted({r["size"] for results in experiments.values() for r in results})
        ax.set_xticks(all_xs)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        if len(experiments) > 1:
            ax.legend(fontsize=8)

    # hide unused axes
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    exp_names = "_vs_".join(experiments.keys())
    fig.suptitle(f"Detection metrics vs training size\n{exp_names}", fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = Path("metrics_vs_size.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.show()


def main(run_dirs: list[Path]) -> None:
    experiments: dict[str, list[dict]] = {}

    for run_dir in run_dirs:
        print(f"\n=== {run_dir.name} ===")
        results = load_results(run_dir)

        needs_f1 = any("f1_optimal" not in r for r in results)
        if needs_f1:
            print("F1/P/R metrics not found in results.json — running val()…")
            try:
                results, changed = enrich_with_f1(run_dir, results)
                if changed:
                    save_results(run_dir, results)
                    print("  Updated results.json with F1/P/R metrics.")
            except ImportError:
                print("  ultralytics not available; skipping F1/P/R enrichment.")

        experiments[run_dir.name] = results

    plot_experiments(experiments)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dirs = [Path(p) for p in sys.argv[1:]]
    else:
        run_dirs = [Path("experiments/sq_c30_m15_col")]

    main(run_dirs)
