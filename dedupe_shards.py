#!/usr/bin/env python3
"""
HouseBrain Cross‑Shard Deduplicator

Ensures global uniqueness of samples across multiple shard folders by hashing
each sample's `input` section. Duplicates (by hash) are removed or moved.

Usage examples:

  # Dry run over all shards under a parent directory
  python dedupe_shards.py --roots /content/HouseBrainLLM --glob "hb_r1_shard_*" --dry-run

  # Enforce dedupe (delete duplicates) under an explicit shards parent
  python dedupe_shards.py --roots /content/HouseBrainLLM --glob "hb_r1_shard_*" --action delete

  # Move duplicates instead of deleting (for audit)
  python dedupe_shards.py --roots /content/HouseBrainLLM --glob "hb_r1_shard_*" --action move --move-dir duplicates_removed
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple, Set, List


def compute_input_hash(sample_path: Path) -> Tuple[str, bool]:
    """Return (hash, ok) for a JSON sample file based on its `input` field."""
    try:
        with sample_path.open("r") as f:
            data = json.load(f)
        if "input" not in data:
            return "", False
        payload = json.dumps(data["input"], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), True
    except Exception:
        return "", False


def iter_sample_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for sub in ("train", "validation"):
        d = root / sub
        if d.exists():
            files.extend(sorted(d.rglob("*.json")))
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def dedupe_shards(roots: List[Path], pattern: str, action: str, move_dir_name: str, dry_run: bool) -> Dict[str, int]:
    """Scan shards under roots, removing/moving duplicates by input hash."""
    seen: Set[str] = set()
    stats = {"scanned": 0, "unique": 0, "dupes": 0, "moved": 0, "deleted": 0, "skipped": 0}

    for root in roots:
        # Find shard directories under root matching the glob pattern
        shards = sorted([p for p in root.glob(pattern) if p.is_dir()])
        for shard in shards:
            for fp in iter_sample_files(shard):
                stats["scanned"] += 1
                h, ok = compute_input_hash(fp)
                if not ok:
                    stats["skipped"] += 1
                    continue
                if h in seen:
                    stats["dupes"] += 1
                    if dry_run:
                        continue
                    if action == "delete":
                        try:
                            fp.unlink(missing_ok=True)
                            stats["deleted"] += 1
                        except Exception:
                            pass
                    elif action == "move":
                        try:
                            rel = fp.relative_to(shard)
                            dst = shard / move_dir_name / rel
                            ensure_dir(dst.parent)
                            shutil.move(str(fp), str(dst))
                            stats["moved"] += 1
                        except Exception:
                            pass
                else:
                    seen.add(h)
                    stats["unique"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross‑shard dedupe by hashing sample['input']")
    parser.add_argument("--roots", nargs="+", required=True, help="Root directories that contain shard folders")
    parser.add_argument("--glob", default="hb_r1_shard_*", help="Glob pattern to match shard directories under each root")
    parser.add_argument("--action", choices=["delete", "move"], default="delete", help="What to do with duplicates")
    parser.add_argument("--move-dir", default="duplicates_removed", help="Subdir under each shard to move duplicates into (when action=move)")
    parser.add_argument("--dry-run", action="store_true", help="Scan only; do not modify files")
    args = parser.parse_args()

    roots = [Path(p).resolve() for p in args.roots]
    stats = dedupe_shards(roots, args.glob, args.action, args.move_dir, args.dry_run)

    print(
        f"Scanned: {stats['scanned']} | Unique: {stats['unique']} | Duplicates: {stats['dupes']} | "
        f"Deleted: {stats['deleted']} | Moved: {stats['moved']} | Skipped: {stats['skipped']}"
    )


if __name__ == "__main__":
    main()\n