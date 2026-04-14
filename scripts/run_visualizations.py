"""
Visualize model predictions for all runs, ordered best to worst by val F2.

Run one arch per GPU simultaneously on kahan:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_visualizations.py --arch deeplabv3plus
  CUDA_VISIBLE_DEVICES=1 uv run python scripts/run_visualizations.py --arch unet

Options:
  --arch          unet | deeplabv3plus | all  (default: all)
  --top-n         Limit to top N runs by best val F2
  --min-f2        Min best_val_f2 % to include (default: 25)
  --splits        Splits to run (default: val test train)
  --dry-run       Print commands without executing
  --skip-existing Skip runs where the 0.5-threshold prob TIF already exists
  --f2-only       Skip 0.5 pass; only run error-only at F2 threshold (requires
                  existing prob TIF from a previous run)

For each eligible run this script does:
  1. Full inference at threshold 0.5  → {split}_pred_s{stride}.tif
                                        {split}_error_s{stride}.tif
  2. Error-only at F2-optimal threshold (if != 0.5)
                                      → {split}_error_s{stride}_f2.tif  (cheap, reuses prob TIF)

F2 thresholds are read from runs_kahan_final_final_export.csv (f2_threshold column).
For runs without that data, they are computed on-the-fly from pr_curves.npz.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

# Shared root: use vacant_lot's _get_shared_root() which handles both
# worktree layout (local: .../Vacant_Lot_Detection/main/scripts/) and
# flat layout (kahan: .../Vacant_Lot_Detection/scripts/).
try:
    from vacant_lot.config import _get_shared_root
    SHARED_ROOT = _get_shared_root()
except Exception:
    # Fallback: walk up until we find outputs/models
    p = SCRIPT_DIR.parent
    while p != p.parent:
        if (p / "outputs" / "models").exists():
            SHARED_ROOT = p
            break
        p = p.parent
    else:
        SHARED_ROOT = SCRIPT_DIR.parent

# train/ scripts live in the worktree (local) or directly in shared root (kahan)
WORKTREE = SHARED_ROOT / "main" if (SHARED_ROOT / "main").exists() else SHARED_ROOT

RUNS_CSV_CANDIDATES = [
    SHARED_ROOT / "runs_kahan_final_final_export.csv",
    SHARED_ROOT / "runs_kahan_final.csv",
    SHARED_ROOT / "runs_kahan_3.csv",
    SHARED_ROOT / "runs_kahan_2.csv",
    SHARED_ROOT / "runs_kahan.csv",
    SHARED_ROOT / "runs.csv",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_runs_csv() -> Path:
    for p in RUNS_CSV_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"No runs CSV found in {SHARED_ROOT}")


def run_dir_path(arch: str, run_id: str) -> Path:
    """Return absolute path to run directory: .../outputs/models/{arch}/kahan_{NNN}"""
    try:
        return SHARED_ROOT / "outputs" / "models" / arch / f"kahan_{int(run_id):03d}"
    except ValueError:
        return SHARED_ROOT / "outputs" / "models" / arch / run_id


def run_rel(arch: str, run_id: str) -> str:
    """Return run path relative to shared root (as passed to --run)."""
    try:
        return f"outputs/models/{arch}/kahan_{int(run_id):03d}"
    except ValueError:
        return f"outputs/models/{arch}/{run_id}"


def parse_threshold(raw: str) -> float | None:
    """
    Parse a threshold value that may be stored as decimal (0.298) or
    accidentally as a percentage (36.6 → 0.366). Returns None if invalid.
    """
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    if 0.0 < v <= 1.0:
        return v
    if 1.0 < v <= 100.0:
        # Entered as percentage — convert silently
        return v / 100.0
    return None   # > 100 or <= 0: bad data


def compute_f2_threshold_from_npz(rdir: Path) -> float | None:
    """Compute F2-optimal threshold from val PR curve in pr_curves.npz."""
    import numpy as np
    pr_path = rdir / "pr_curves.npz"
    if not pr_path.exists():
        return None
    try:
        d = np.load(pr_path)
        if "val_pr_precision" not in d:
            return None
        prec  = d["val_pr_precision"][:-1]   # sklearn appends a trailing point
        rec   = d["val_pr_recall"][:-1]
        thresh = d["val_pr_thresholds"]
        f2    = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-8)
        return float(thresh[np.argmax(f2)])
    except Exception as e:
        print(f"    [warn] could not compute F2 threshold from NPZ: {e}", file=sys.stderr)
        return None


def parse_best_val_f2(row: dict) -> float:
    """Return best_val_f2 as a fraction (0–1). Falls back to val_f2 at 0.5 threshold."""
    for key in ("best_val_f2", "val_f2"):
        raw = row.get(key, "")
        if not raw:
            continue
        try:
            v = float(raw)
            return v / 100.0 if v > 1.0 else v
        except ValueError:
            continue
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch visualize predictions, best to worst by F2")
    parser.add_argument("--runs-csv", default=None, help="Explicit path to runs CSV (overrides auto-detect)")
    parser.add_argument("--arch", default="all", choices=["unet", "deeplabv3plus", "all"])
    parser.add_argument("--top-n", type=int, default=None, help="Limit to top N runs")
    parser.add_argument("--min-f2", type=float, default=25.0, help="Min best_val_f2 %% to include")
    parser.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip pass 1 if prob TIF already exists")
    parser.add_argument("--f2-only", action="store_true",
                        help="Only run error-only at F2 threshold (requires existing prob TIF)")
    args = parser.parse_args()

    runs_csv = Path(args.runs_csv) if args.runs_csv else find_runs_csv()
    print(f"Shared root: {SHARED_ROOT}")
    print(f"Worktree   : {WORKTREE}")
    print(f"Runs CSV   : {runs_csv}")
    print(f"Splits   : {args.splits}")

    with open(runs_csv) as f:
        rows = list(csv.DictReader(f))

    # Filter by arch
    if args.arch != "all":
        rows = [r for r in rows if r["model_type"] == args.arch]

    # Attach sort key and filter by min F2
    for r in rows:
        r["_sort_f2"] = parse_best_val_f2(r)
    rows = [r for r in rows if r["_sort_f2"] * 100 >= args.min_f2]

    # Sort best to worst
    rows.sort(key=lambda r: r["_sort_f2"], reverse=True)

    # Top N
    if args.top_n:
        rows = rows[:args.top_n]

    # Preview
    print(f"\n{'Arch':15s} {'Run':>5s} {'Patch':>6s} {'BestValF2':>10s} {'F2Thr':>7s}")
    print("-" * 50)
    for r in rows:
        print(f"  {r['model_type']:15s} {r['run_id']:>5} "
              f"{r.get('patch_size') or '?':>6}  "
              f"{r['_sort_f2']*100:>8.1f}%  "
              f"{r.get('f2_threshold','—'):>7}")
    print()

    if args.dry_run:
        print("[dry-run mode — commands printed but not executed]\n")

    # -----------------------------------------------------------------------
    # Process each run
    # -----------------------------------------------------------------------
    for r in rows:
        arch     = r["model_type"]
        run_id   = r["run_id"]
        rdir     = run_dir_path(arch, run_id)
        rel      = run_rel(arch, run_id)
        patch_sz = int(r.get("patch_size") or 512)

        print(f"{'='*65}")
        print(f"[{arch}  run {run_id}]  patch={patch_sz}  dir={rdir.name}")

        if not rdir.exists():
            print(f"  [SKIP] directory not found: {rdir}")
            continue

        # Stride / batch size
        if patch_sz >= 1024:
            stride, bs = 512, 2
        elif patch_sz >= 512:
            stride, bs = 256, 4
        else:
            stride, bs = patch_sz // 2, 8

        # F2 threshold — CSV first, then pr_curves.npz fallback
        f2_thr = parse_threshold(r.get("f2_threshold", ""))
        if f2_thr is None:
            print(f"  No valid f2_threshold in CSV — computing from pr_curves.npz")
            f2_thr = compute_f2_threshold_from_npz(rdir)
            if f2_thr is not None:
                print(f"  Computed F2 threshold: {f2_thr:.4f}")

        # ---- Pass 1: full inference @ 0.5 --------------------------------
        if not args.f2_only:
            prob_tif = rdir / "figures" / f"val_pred_s{stride}.tif"
            if args.skip_existing and prob_tif.exists():
                print(f"  [skip pass 1] prob TIF already exists")
            else:
                cmd = [
                    "uv", "run", "python", "train/visualize_predictions.py",
                    "--run", rel,
                    "--threshold", "0.5",
                    "--stride", str(stride),
                    "--batch-size", str(bs),
                    "--splits", *args.splits,
                ]
                print(f"  Pass 1 @ 0.5")
                print(f"    {' '.join(cmd)}")
                if not args.dry_run:
                    subprocess.run(cmd, cwd=WORKTREE, check=True)

        # ---- Pass 2: error-only @ F2 threshold ----------------------------
        if f2_thr is None:
            print(f"  [skip pass 2] no F2 threshold")
        elif abs(f2_thr - 0.5) < 0.01:
            print(f"  [skip pass 2] F2 threshold {f2_thr:.4f} ≈ 0.5 — single pass sufficient")
        else:
            # Encode threshold in suffix: 0.298 → _t298, 0.355 → _t355
            thr_tag = f"_t{int(round(f2_thr * 1000))}"
            cmd2 = [
                "uv", "run", "python", "train/visualize_predictions.py",
                "--run", rel,
                "--threshold", f"{f2_thr:.4f}",
                "--stride", str(stride),
                "--error-only",
                "--suffix", thr_tag,
                "--splits", *args.splits,
            ]
            print(f"  Pass 2 (error-only) @ {f2_thr:.4f}  suffix={thr_tag}")
            print(f"    {' '.join(cmd2)}")
            if not args.dry_run:
                subprocess.run(cmd2, cwd=WORKTREE, check=True)

    print(f"\n{'='*65}")
    print("Done.")


if __name__ == "__main__":
    main()
