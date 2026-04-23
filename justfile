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
