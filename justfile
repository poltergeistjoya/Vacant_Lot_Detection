# Vacant Lot Detection — task runner
# Usage: just <recipe>

mod data-prep 'data_prep/justfile'
mod train 'train/justfile'

default:
    @just --list --list-submodules

upload-kaggle *ARGS:
    uv run python scripts/upload_kaggle_dataset.py {{ARGS}}
