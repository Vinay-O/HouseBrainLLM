#!/usr/bin/env python3
from __future__ import annotations

import tarfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    training = root / "training_dataset"
    archive_dir = root / "data" / "training" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if training.exists() and training.is_dir():
        tar_path = archive_dir / "training_dataset_backup.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(training, arcname="training_dataset")
        print(f"✅ Archived training_dataset to {tar_path}")
    else:
        print("ℹ️ training_dataset not found; nothing to archive")


if __name__ == "__main__":
    main()
