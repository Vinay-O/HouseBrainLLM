"""
Mock inference that returns curated examples by scenario id, simulating the LLM API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


EXAMPLES_DIR = Path("examples")


def load_scenario(sid: str) -> Dict[str, Any]:
    """Load a curated example by id: e.g., 's01', 's12'."""
    if not EXAMPLES_DIR.exists():
        raise FileNotFoundError("examples/ directory is missing. Run create_curated_examples.py")
    # Prefer exact match file
    for p in sorted(EXAMPLES_DIR.glob(f"{sid}_*.json")):
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback to first example
    p = sorted(EXAMPLES_DIR.glob("*.json"))[0]
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_housebrain_output(input_payload: Dict[str, Any], scenario_id: str = "s01") -> Dict[str, Any]:
    """Return a curated example; in the future, this calls the trained model."""
    ex = load_scenario(scenario_id)
    # Replace input with provided payload so downstream sees the same interface
    ex["input"] = input_payload
    return ex


