# Vacant Lot Detection — task runner
# Usage: just <recipe>

mod data-prep 'data_prep/justfile'
mod train 'train/justfile'

default:
    @just --list --list-submodules

upload-kaggle *ARGS:
    uv run python scripts/upload_kaggle_dataset.py {{ARGS}}

# Parcel-level confusion analysis: TP/FP/FN/TN by land use and building class
# Usage:
#   just eval-confusion outputs/models/deeplabv3plus/kahan_027
#   just eval-confusion outputs/models/deeplabv3plus/kahan_027 test
#   just eval-confusion outputs/models/deeplabv3plus/kahan_027 val --out results.csv --plot-coverage 0.3
#   just eval-confusion outputs/models/deeplabv3plus/kahan_027 val --nonvacant-only
eval-confusion RUN SPLIT="val" *ARGS:
    uv run python scripts/eval_confusion_by_class.py --run {{RUN}} --split {{SPLIT}} {{ARGS}}

# Export confusion results to GeoPackage for QGIS visualization
# Usage:
#   just export-confusion outputs/models/deeplabv3plus/kahan_027                      # default: {run}/confusion_parcels_val.gpkg
#   just export-confusion outputs/models/deeplabv3plus/kahan_027 --out custom.gpkg    # custom output name (in run dir)
#   just export-confusion outputs/models/deeplabv3plus/kahan_027 --split test         # test split
#   just export-confusion outputs/models/deeplabv3plus/kahan_027 --coverage 0.3
export-confusion RUN *ARGS:
    uv run python scripts/export_confusion_to_gpkg.py --run {{RUN}} {{ARGS}}

# Justify threshold choice: quantitative analysis of precision/recall/F2 across thresholds
# Usage:
#   just justify-threshold outputs/models/deeplabv3plus/kahan_027
#   just justify-threshold outputs/models/deeplabv3plus/kahan_027 --split test
justify-threshold RUN *ARGS:
    uv run python scripts/justify_threshold.py --run {{RUN}} {{ARGS}}

# Full DL evaluation pipeline: plot results → predict test split → error map at F2-optimal threshold
# Reads eval_stride from run's config.yaml; falls back to patch_size // 2 if absent.
# Usage:
#   just eval-dl outputs/models/unet/kahan_041
eval-dl RUN:
    #!/usr/bin/env bash
    set -euo pipefail
    stride=$(uv run python -c "
import yaml
try:
    cfg = yaml.safe_load(open('../{{RUN}}/config.yaml'))
    s = cfg.get('eval_stride')
    print('' if s is None else s, end='')
except Exception:
    print('', end='')
")
    stride_arg=$( [ -n "$stride" ] && echo "--stride $stride" || echo "" )
    threshold=$(uv run python train/plot_results.py --run {{RUN}} | tee /dev/stderr | grep "F2-optimal threshold" | awk '{print $NF}')
    echo ">>> eval-dl: threshold=$threshold  stride=${stride:-default (patch_size // 2)}"
    uv run python train/visualize_predictions.py --run {{RUN}} --splits test $stride_arg
    uv run python train/visualize_predictions.py --run {{RUN}} --error-only --splits test --threshold "$threshold" $stride_arg
