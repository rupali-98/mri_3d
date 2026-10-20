"""Default locations and the shared command-line parser.

Where the dataset lives depends on the machine, not the experiment, so it is
kept out of configs/*.yaml. Every script resolves it the same way, highest
priority first:

    1. command line        --data-dir /path/to/data
    2. environment         HIPPO_DATA_DIR=/path/to/data
    3. config file         data-dir = /path/to/data   in ./hippo.conf or ~/.hippo.conf
    4. DATA_DIR below
"""
from __future__ import annotations

from pathlib import Path

import configargparse

DATA_DIR = "data"                     # holds Task04_Hippocampus/ and splits.json
SPLIT_FILENAME = "splits.json"
CONFIG = "configs/baseline.yaml"
RUN_DIR = "runs/baseline"

# Later files override earlier ones.
CONFIG_FILES = ["~/.hippo.conf", "hippo.conf"]


def make_parser(description: str | None = None) -> configargparse.ArgParser:
    """Parser with the options every hippo script shares."""
    p = configargparse.ArgParser(
        description=description,
        default_config_files=CONFIG_FILES,
        # Lets hippo.conf hold keys only some scripts use (e.g. `run` for evaluate).
        ignore_unknown_config_file_keys=True,
    )
    p.add_argument("--settings", is_config_file=True, help="Extra settings file (same format as hippo.conf)")
    p.add_argument("--data-dir", env_var="HIPPO_DATA_DIR", default=DATA_DIR,
                   help="Folder containing Task04_Hippocampus/ and splits.json")
    return p


def find_case(name: str, data_dir: str) -> tuple[Path, Path | None]:
    """Locate a case by name in the MSD folders; labels exist only for imagesTr."""
    task = Path(data_dir) / "Task04_Hippocampus"
    file = name if name.endswith(".nii.gz") else f"{name}.nii.gz"
    for images, labels in [("imagesTr", "labelsTr"), ("imagesTs", None)]:
        if (task / images / file).exists():
            label = task / labels / file if labels else None
            return task / images / file, label
    raise SystemExit(f"{file} not found in {task}/imagesTr or imagesTs")
