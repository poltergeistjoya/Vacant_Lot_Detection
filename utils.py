from pathlib import Path
import os
from logger import get_logger

log = get_logger()

def create_output_dir(output_dir: str | Path) -> Path:
    """
    Create output directory if it doesn't exist.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"📁 Created output directory: {output_dir}")
    else:
        log.info(f"📂 Using existing output directory: {output_dir}")
    return output_dir
