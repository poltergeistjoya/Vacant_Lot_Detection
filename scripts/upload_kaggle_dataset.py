"""
Create or update a Kaggle dataset with segmentation supplementary data.

Copies the specified files into a staging directory, writes dataset-metadata.json,
then calls the Kaggle API to create a new dataset or push a new version.

Usage:
    # Create new dataset
    uv run python scripts/upload_kaggle_dataset.py \\
        --files /path/to/vacancy_mask.tif /path/to/patch_splits.json \\
        --dataset-id poltergeistjoya/nyc-vacancy-seg-masks \\
        --title "NYC Vacancy Segmentation Masks" \\
        --new

    # Push new version to existing dataset
    uv run python scripts/upload_kaggle_dataset.py \\
        --files /path/to/vacancy_mask.tif /path/to/patch_splits.json \\
        --dataset-id poltergeistjoya/nyc-vacancy-seg-masks \\
        --message "Updated patch splits with 2022 TIGER/Line borough boundaries"
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from kaggle import KaggleApi  # type: ignore

from vacant_lot.logger import get_logger

log = get_logger()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create or update a Kaggle dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--files",
        nargs="+",
        required=True,
        metavar="FILE",
        help="One or more file paths to include in the dataset.",
    )
    p.add_argument(
        "--dataset-id",
        required=True,
        metavar="USERNAME/SLUG",
        help="Kaggle dataset ID, e.g. poltergeistjoya/nyc-vacancy-seg-masks",
    )
    p.add_argument(
        "--title",
        default=None,
        metavar="TITLE",
        help="Human-readable dataset title (required when --new is set).",
    )
    p.add_argument(
        "--message",
        default="New version",
        metavar="MSG",
        help="Version message shown in Kaggle history (default: 'New version').",
    )
    p.add_argument(
        "--new",
        action="store_true",
        help="Create a new dataset instead of pushing a new version.",
    )
    p.add_argument(
        "--staging-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to stage files before upload. "
            "Defaults to a temp dir that is cleaned up after upload."
        ),
    )
    return p


def write_metadata(staging_dir: Path, dataset_id: str, title: str) -> None:
    username, slug = dataset_id.split("/", 1)
    meta = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": [],
        "collaborators": [],
        "data": [],
    }
    meta_path = staging_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info(f"Wrote metadata: {meta_path}")


def stage_files(files: list[str], staging_dir: Path) -> None:
    for src in files:
        src_path = Path(src)
        if not src_path.exists():
            log.error(f"File not found: {src_path}")
            sys.exit(1)
        dst = staging_dir / src_path.name
        shutil.copy2(src_path, dst)
        log.info(f"Staged: {src_path.name} ({src_path.stat().st_size / 1_048_576:.1f} MB)")


def main() -> None:
    args = build_parser().parse_args()

    # Validate dataset-id format
    if "/" not in args.dataset_id:
        log.error("--dataset-id must be in the format username/slug")
        sys.exit(1)

    if args.new and not args.title:
        log.error("--title is required when creating a new dataset (--new)")
        sys.exit(1)

    title = args.title or args.dataset_id.split("/", 1)[1].replace("-", " ").title()

    # Set up staging directory
    cleanup = args.staging_dir is None
    staging_dir = Path(args.staging_dir) if args.staging_dir else Path(tempfile.mkdtemp())
    staging_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Staging directory: {staging_dir}")

    try:
        stage_files(args.files, staging_dir)
        write_metadata(staging_dir, args.dataset_id, title)

        api = KaggleApi()
        api.authenticate()

        if args.new:
            log.info(f"Creating new dataset: {args.dataset_id}")
            api.dataset_create_new(
                folder=str(staging_dir),
                public=True,
                quiet=False,
                convert_to_csv=False,
                dir_mode="zip",
            )
            log.info(f"Dataset created: https://www.kaggle.com/datasets/{args.dataset_id}")
        else:
            log.info(f"Pushing new version to: {args.dataset_id}")
            api.dataset_create_version(
                folder=str(staging_dir),
                version_notes=args.message,
                quiet=False,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="zip",
            )
            log.info(f"Version pushed: https://www.kaggle.com/datasets/{args.dataset_id}")

    finally:
        if cleanup:
            shutil.rmtree(staging_dir)
            log.info("Cleaned up staging directory")


if __name__ == "__main__":
    main()
