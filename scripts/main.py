"""Train YOLO on increasingly large subsets of the dataset and compare metrics.

Run with: `python scripts/main.py` (from project root, inside an active venv).
Pick which experiment to run by changing the last line: run(test1) / run(test2) / run(test3).
"""

import json
import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from dotenv import load_dotenv
from ultralytics import YOLO

from data_generation.dataset_synthetic import load_synthetic_rails
from data_generation.dataset_synthetic_square import load_synthetic_rails_square
from data_generation.make_greyscale import convert as convert_to_greyscale

load_dotenv()

# ── Shared training constants ─────────────────────────────────────────────────
EPOCHS     = 25
IMGSZ      = 640
DEVICE     = "mps"       # Apple Silicon GPU; use "cpu" or "0" (CUDA) on other machines
SEED       = 42
WEIGHTS    = "models/yolo11n.pt"
TEST_RATIO = 0.2


# ── Experiment configs ────────────────────────────────────────────────────────

@dataclass
class Config:
    run_name:    str
    use_square:  bool        # True = 640×640 square; False = 570×100 rectangular
    use_grey:    bool        # True = convert colour images to greyscale before training
    syn_config:  str         # key into CONFIGS in dataset_synthetic*.py
    n_samples:   int
    subset_sizes: list[int]
    data_source: Path | None = None  # None = generate fresh; Path = reuse existing dataset
    epochs:      int = EPOCHS
    extra_train_kwargs: dict = None  # extra kwargs forwarded to model.train()


def test1() -> Config:
    """Main run — full learning curve on square colour images."""
    return Config(
        run_name     = "sq_c30_m15_col",
        use_square   = True,
        use_grey     = False,
        syn_config   = "c30_m15",   # p_clip=0.30, p_motif=0.15
        n_samples    = 4000,
        subset_sizes = [100, 200, 400, 800, 1600, 2800],
        data_source  = None,
    )


def test2() -> Config:
    """Spot-check — rectangular (cropped) images, two sizes only."""
    return Config(
        run_name     = "rect_c30_m15_col",
        use_square   = False,
        use_grey     = False,
        syn_config   = "c30_m15",
        n_samples    = 4000,
        subset_sizes = [200, 800],
        data_source  = None,
    )


def test3() -> Config:
    """Spot-check — greyscale square images, reuses test1's generated data."""
    return Config(
        run_name     = "sq_c30_m15_grey",
        use_square   = True,
        use_grey     = True,
        syn_config   = "c30_m15",
        n_samples    = 4000,
        subset_sizes = [200, 800],
        data_source  = Path("experiments/sq_c30_m15_col/dataset/c30_m15"),
    )


def test4() -> Config:
    """Convergence check — same 1600 images as test1, more epochs to test underfitting hypothesis."""
    return Config(
        run_name     = "sq_c30_m15_col_ep50",
        use_square   = True,
        use_grey     = False,
        syn_config   = "c30_m15",
        n_samples    = 4000,
        subset_sizes = [1600],
        data_source  = Path("experiments/sq_c30_m15_col/dataset/c30_m15"),
        epochs       = 50,
    )


def test5() -> Config:
    """LR elasticity check — same sizes as test1 but only 8 epochs.

    If the drop from 800→2800 is driven by the cosine LR schedule decaying before
    the model converges on larger sets, all sizes should perform similarly here
    (LR still high, little difference in effective steps per epoch relative to warmup).
    Extra data without extra variance should show no benefit at all.
    """
    return Config(
        run_name     = "sq_c30_m15_col_ep8",
        use_square   = True,
        use_grey     = False,
        syn_config   = "c30_m15",
        n_samples    = 4000,
        subset_sizes = [100, 200, 400, 800, 1600, 2800],
        data_source  = Path("experiments/sq_c30_m15_col/dataset/c30_m15"),
        epochs       = 8,
    )


def test6() -> Config:
    """Geometric-augmentation ablation — same conditions as test1, sizes 800 and 1600 only.

    All ultralytics geometric augmentations are disabled so the model only sees
    colour/HSV jitter. Goal: isolate whether geometric augmentation helps or hurts
    on these synthetically-rendered images where geometry is already varied at
    generation time.
    """
    return Config(
        run_name     = "sq_c30_m15_col_nogeom",
        use_square   = True,
        use_grey     = False,
        syn_config   = "c30_m15",
        n_samples    = 4000,
        subset_sizes = [800, 1600],
        data_source  = Path("experiments/sq_c30_m15_col/dataset/c30_m15"),
        extra_train_kwargs = {
            # Disable all geometric transforms; keep colour augmentations at defaults
            "degrees":     0.0,
            "translate":   0.0,
            "scale":       0.0,
            "shear":       0.0,
            "perspective": 0.0,
            "flipud":      0.0,
            "fliplr":      0.0,
            "mosaic":      0.0,
            "mixup":       0.0,
            "copy_paste":  0.0,
        },
    )


# ── Training logic ────────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    run_dir      = Path("experiments") / cfg.run_name
    work_dir     = run_dir / "splits"
    results_file = run_dir / "results.json"
    runs_dir     = run_dir / "detect"

    loader = load_synthetic_rails_square if cfg.use_square else load_synthetic_rails
    load_dataset = partial(loader, config=cfg.syn_config, n_samples=cfg.n_samples)

    # ── 1. Prepare the dataset ────────────────────────────────────────────────
    run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = cfg.data_source if cfg.data_source is not None else run_dir / "dataset"
    if cfg.data_source is not None:
        info = {
            "images_dir": cfg.data_source / "images",
            "labels_dir": cfg.data_source / "labels",
            "classes": ["crocodile_clip"],
        }
    else:
        info = load_dataset(data_dir)

    if cfg.use_grey:
        grey_dir = data_dir.parent / (data_dir.name + "_grey")
        convert_to_greyscale(data_dir, grey_dir)
        info = {**info, "images_dir": grey_dir / "images"}

    work_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Build a fixed train/test split ─────────────────────────────────────
    images = sorted(
        p for p in info["images_dir"].iterdir()
        if (info["labels_dir"] / (p.stem + ".txt")).exists()
    )
    random.Random(SEED).shuffle(images)

    n_test     = int(len(images) * TEST_RATIO)
    test_set   = images[:n_test]
    train_pool = images[n_test:]
    print(f"Total: {len(images)} | Test: {len(test_set)} | Train pool: {len(train_pool)}")

    test_file = work_dir / "test.txt"
    test_file.write_text("\n".join(str(p.resolve()) for p in test_set))

    # ── 3. Train once per subset size ─────────────────────────────────────────
    results = []
    for size in cfg.subset_sizes:
        if size > len(train_pool):
            print(f"Skipping size={size}: only {len(train_pool)} train images available")
            continue

        train_file = work_dir / f"train_{size}.txt"
        train_file.write_text("\n".join(str(p.resolve()) for p in train_pool[:size]))

        yaml_path = work_dir / f"dataset_{size}.yaml"
        yaml_path.write_text(yaml.dump({
            "path": str(work_dir.resolve()),
            "train": train_file.name,
            "val":   test_file.name,
            "names": {i: c for i, c in enumerate(info["classes"])},
        }))

        print(f"\n=== Training with {size} images ===")
        model = YOLO(WEIGHTS)
        model.train(
            data     = str(yaml_path),
            epochs   = cfg.epochs,
            imgsz    = IMGSZ,
            device   = DEVICE,
            project  = str(runs_dir.resolve()),
            name     = f"size_{size}",
            exist_ok = True,
            seed     = SEED,
            rect     = not cfg.use_square,
            **(cfg.extra_train_kwargs or {}),
        )

        metrics = model.val(
            data     = str(yaml_path),
            device   = DEVICE,
            project  = str(runs_dir.resolve()),
            name     = f"size_{size}_val",
            exist_ok = True,
        )
        results.append({
            "size":         size,
            "mAP@.5":       float(metrics.box.map50),
            "mAP@.5:.95":   float(metrics.box.map),
            "inference_ms": float(metrics.speed.get("inference", 0.0)),
        })

        results_file.write_text(json.dumps(results, indent=2))

    # ── 4. Print summary table ────────────────────────────────────────────────
    print(f"\n{'Size':>6} {'mAP@.5':>10} {'mAP@.5:.95':>12} {'Inf (ms)':>10}")
    for r in results:
        print(f"{r['size']:>6} {r['mAP@.5']:>10.4f} {r['mAP@.5:.95']:>12.4f} {r['inference_ms']:>10.2f}")


if __name__ == "__main__":
    run(test6())   # ← change to test1() / test2() / test3() / test4() / test5() / test6() to switch experiments
