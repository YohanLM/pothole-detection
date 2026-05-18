"""Analyse whether model mistakes correlate with clip-like motifs.

Two-step approach, both cacheable:
  1. Replay the dataset RNG (seed=42) to record which images have hard-negative
     motifs, without re-saving any images.
  2. Run YOLO inference on the test split with the chosen model and compute
     per-image TP/FP/TN/FN.

Then stratify errors by motif presence and plot.

Usage:
    python analyse_motif_errors.py                      # default: size_2800 model
    python analyse_motif_errors.py --size 800           # choose training size
    python analyse_motif_errors.py --conf 0.25          # confidence threshold
    python analyse_motif_errors.py --refresh-metadata   # force RNG replay
    python analyse_motif_errors.py --refresh-preds      # force re-inference
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent
EXP_DIR     = REPO_ROOT / "experiments" / "sq_c30_m15_col"
DATASET_DIR = EXP_DIR / "dataset" / "c30_m15"
SPLITS_DIR  = EXP_DIR / "splits"
DETECT_DIR  = EXP_DIR / "detect"
META_CACHE  = EXP_DIR / "metadata.json"
PRED_CACHE  = EXP_DIR / "predictions_{size}.json"

N_SAMPLES = 4000
SEED      = 42
DEVICE    = "mps"
IOU_THR   = 0.5      # IoU threshold to count a detection as a true positive


# ── Step 1: metadata via RNG replay ───────────────────────────────────────────

def _build_inter_rail_tracked(ds):
    """Return a replacement for ds._add_inter_rail_features that records the
    motif draw. We replicate the function's logic exactly; the only addition
    is capturing the boolean before/after the p_motif gate."""
    _flag = [False]

    def _tracked(rng, img, rails, p_motif=0.05):
        h, w = img.shape[:2]
        for top_rail, bot_rail in [(rails[0], rails[1]), (rails[2], rails[3])]:
            gap_y0, gap_y1 = ds._gap(top_rail, bot_rail)
            if gap_y1 - gap_y0 < 6:
                continue
            green_cx = int(rng.integers(10, w - 10))
            green_cy = int(rng.integers(gap_y0 + 2, gap_y1 - 2))
            green_r  = int(rng.integers(2, 4))
            ds._draw_blob(rng, img, green_cx, green_cy, green_r, ds.GREEN_RGB)
            n_dots = int(rng.integers(2, 7))
            for _ in range(n_dots):
                dx = int(rng.integers(0, w))
                dy = int(rng.integers(gap_y0 + 1, gap_y1 - 1))
                dr = int(rng.integers(1, 3))
                colour = ds.GREEN_RGB if rng.random() < 0.5 else ds.RED_NOISE_RGB
                ds._draw_blob(rng, img, dx, dy, dr, colour)

        drew_motif = rng.random() < p_motif
        _flag[0] = drew_motif
        if drew_motif:
            top_rail, bot_rail = (
                (rails[0], rails[1]) if rng.random() < 0.5 else (rails[2], rails[3])
            )
            gap_y0, gap_y1 = ds._gap(top_rail, bot_rail)
            if gap_y1 - gap_y0 >= 6:
                motif_type = "A" if rng.random() < 0.5 else "B"
                drawer = (
                    ds._draw_motif_red_green_tail
                    if motif_type == "A"
                    else ds._draw_motif_wide_with_gap
                )
                drawer(rng, img, gap_y0, gap_y1)
                drawer(rng, img, gap_y0, gap_y1)

    return _tracked, _flag


def build_metadata(force: bool = False) -> dict[str, dict]:
    """Replay the RNG for all N_SAMPLES images and record has_clip / has_motif.

    Result is cached in META_CACHE. Keys are image stems ('rail_00042').
    """
    if META_CACHE.exists() and not force:
        print(f"Loading cached metadata from {META_CACHE}")
        return json.loads(META_CACHE.read_text())

    print("Replaying RNG to build image metadata (no images written)…")
    sys.path.insert(0, str(REPO_ROOT / "data_generation"))
    import dataset_synthetic as ds
    from dataset_synthetic_square import CONFIGS, _make_image_square

    cfg = CONFIGS["c30_m15"]
    rng = np.random.default_rng(SEED)

    # Monkey-patch ds._add_inter_rail_features so we can read the motif flag.
    # The square generator calls ds._add_inter_rail_features(rng, img, rails)
    # (no p_motif arg), so the default p_motif=0.05 applies.
    tracked_fn, motif_flag = _build_inter_rail_tracked(ds)
    ds._add_inter_rail_features = tracked_fn

    metadata = {}
    for i in range(N_SAMPLES):
        img, bbox = _make_image_square(rng, cfg)
        stem = f"rail_{i:05d}"
        metadata[stem] = {
            "has_clip":  bbox is not None,
            "has_motif": bool(motif_flag[0]),
        }
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{N_SAMPLES}")

    META_CACHE.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata to {META_CACHE}")
    return metadata


# ── Step 2: inference on the test split ───────────────────────────────────────

def _iou(boxA, boxB):
    """IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def _read_gt_box(label_path: Path, img_w=640, img_h=640):
    """Read YOLO label file → [x1,y1,x2,y2] in pixels, or None."""
    text = label_path.read_text().strip()
    if not text:
        return None
    parts = text.split()
    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1 = (cx - bw / 2) * img_w;  y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w;  y2 = (cy + bh / 2) * img_h
    return [x1, y1, x2, y2]


def run_inference(size: int, conf: float, force: bool = False) -> dict[str, dict]:
    """Run YOLO predict on the test split and cache per-image results."""
    cache_path = Path(str(PRED_CACHE).format(size=size))
    if cache_path.exists() and not force:
        print(f"Loading cached predictions from {cache_path}")
        return json.loads(cache_path.read_text())

    from ultralytics import YOLO

    weight = DETECT_DIR / f"size_{size}" / "weights" / "best.pt"
    if not weight.exists():
        raise FileNotFoundError(f"No weights at {weight}")

    test_images = [
        Path(p) for p in SPLITS_DIR.joinpath("test.txt").read_text().splitlines()
    ]
    print(f"Running inference on {len(test_images)} test images (model size={size})…")
    model = YOLO(str(weight))

    predictions = {}
    results = model.predict(
        [str(p) for p in test_images],
        conf=conf,
        device=DEVICE,
        verbose=False,
        stream=True,
    )
    for img_path, result in zip(test_images, results):
        stem = img_path.stem
        label_path = DATASET_DIR / "labels" / f"{stem}.txt"
        gt_box = _read_gt_box(label_path)

        pred_boxes = result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) else []

        # Determine outcome: if GT has a clip, best IoU with any prediction
        if gt_box is not None:
            best_iou = max((_iou(gt_box, pb) for pb in pred_boxes), default=0.0)
            tp = best_iou >= IOU_THR
            outcome = "TP" if tp else "FN"
        else:
            outcome = "FP" if len(pred_boxes) > 0 else "TN"

        predictions[stem] = {
            "outcome":    outcome,
            "n_preds":    len(pred_boxes),
            "gt_has_clip": gt_box is not None,
        }

    cache_path.write_text(json.dumps(predictions, indent=2))
    print(f"Saved predictions to {cache_path}")
    return predictions


# ── Step 3: analysis & plotting ───────────────────────────────────────────────

def analyse_and_plot(metadata: dict, predictions: dict, size: int) -> None:
    # Join on image stem (only test images appear in predictions)
    rows = []
    for stem, pred in predictions.items():
        if stem not in metadata:
            continue
        rows.append({
            "stem":       stem,
            "has_motif":  metadata[stem]["has_motif"],
            "outcome":    pred["outcome"],
            "gt_has_clip": pred["gt_has_clip"],
        })

    def _subset(rows, motif: bool):
        return [r for r in rows if r["has_motif"] == motif]

    def _counts(rows):
        c = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for r in rows:
            c[r["outcome"]] += 1
        return c

    no_motif = _subset(rows, False)
    with_motif = _subset(rows, True)
    c_no  = _counts(no_motif)
    c_yes = _counts(with_motif)

    def _fp_rate(c):
        denom = c["FP"] + c["TN"]
        return c["FP"] / denom if denom else float("nan")

    def _fn_rate(c):
        denom = c["FN"] + c["TP"]
        return c["FN"] / denom if denom else float("nan")

    def _precision(c):
        denom = c["TP"] + c["FP"]
        return c["TP"] / denom if denom else float("nan")

    def _recall(c):
        denom = c["TP"] + c["FN"]
        return c["TP"] / denom if denom else float("nan")

    # ── Summary print ──────────────────────────────────────────────────────────
    print(f"\n{'':30s} {'No motif':>12} {'With motif':>12}")
    print(f"{'Images':30s} {len(no_motif):>12} {len(with_motif):>12}")
    print(f"{'TP':30s} {c_no['TP']:>12} {c_yes['TP']:>12}")
    print(f"{'FP':30s} {c_no['FP']:>12} {c_yes['FP']:>12}")
    print(f"{'TN':30s} {c_no['TN']:>12} {c_yes['TN']:>12}")
    print(f"{'FN':30s} {c_no['FN']:>12} {c_yes['FN']:>12}")
    print(f"{'FP rate (false alarm)':30s} {_fp_rate(c_no):>12.3f} {_fp_rate(c_yes):>12.3f}")
    print(f"{'FN rate (miss rate)':30s} {_fn_rate(c_no):>12.3f} {_fn_rate(c_yes):>12.3f}")
    print(f"{'Precision':30s} {_precision(c_no):>12.3f} {_precision(c_yes):>12.3f}")
    print(f"{'Recall':30s} {_recall(c_no):>12.3f} {_recall(c_yes):>12.3f}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    labels  = ["No motif", "With motif"]
    c_pairs = [c_no, c_yes]
    sizes   = [len(no_motif), len(with_motif)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(f"Error breakdown by motif presence  (model size={size})", fontsize=13)

    colors_outcome = {"TP": "#4caf50", "TN": "#90caf9", "FP": "#ef5350", "FN": "#ff9800"}

    # Left: stacked bar of outcome counts
    ax = axes[0]
    bottoms = [0, 0]
    for outcome in ["TP", "TN", "FP", "FN"]:
        vals = [c["outcome_count"] if "outcome_count" in c else c.get(outcome, 0)
                for c in c_pairs]
        vals = [c[outcome] for c in c_pairs]
        ax.bar(labels, vals, bottom=bottoms, label=outcome,
               color=colors_outcome[outcome], edgecolor="white", linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title("Outcome counts")
    ax.set_ylabel("Images")
    ax.legend(loc="upper right", fontsize=9)

    # Middle: FP rate and FN rate as grouped bars
    ax = axes[1]
    x = np.arange(2)
    w = 0.35
    fp_rates = [_fp_rate(c) for c in c_pairs]
    fn_rates = [_fn_rate(c) for c in c_pairs]
    ax.bar(x - w/2, fp_rates, w, label="FP rate", color="#ef5350")
    ax.bar(x + w/2, fn_rates, w, label="FN rate (miss)", color="#ff9800")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("Rate")
    ax.set_title("Error rates")
    ax.legend(fontsize=9)
    for i, (fp, fn) in enumerate(zip(fp_rates, fn_rates)):
        ax.text(i - w/2, fp + 0.02, f"{fp:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, fn + 0.02, f"{fn:.2f}", ha="center", fontsize=8)

    # Right: precision & recall
    ax = axes[2]
    precs = [_precision(c) for c in c_pairs]
    recs  = [_recall(c)    for c in c_pairs]
    ax.bar(x - w/2, precs, w, label="Precision", color="#42a5f5")
    ax.bar(x + w/2, recs,  w, label="Recall",    color="#66bb6a")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Precision & Recall")
    ax.legend(fontsize=9)
    for i, (p, r) in enumerate(zip(precs, recs)):
        ax.text(i - w/2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    out = EXP_DIR / f"motif_error_analysis_{size}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out}")
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=2800,
                   help="Training size of the model to evaluate (default: 2800)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold for predictions (default: 0.25)")
    p.add_argument("--refresh-metadata", action="store_true",
                   help="Force RNG replay even if metadata.json exists")
    p.add_argument("--refresh-preds", action="store_true",
                   help="Force re-inference even if prediction cache exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata    = build_metadata(force=args.refresh_metadata)
    predictions = run_inference(args.size, args.conf, force=args.refresh_preds)
    analyse_and_plot(metadata, predictions, args.size)
