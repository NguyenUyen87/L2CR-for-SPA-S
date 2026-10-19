"""Shared paths and helpers for the experiment scripts (run them from the repository root)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"
RESULTS.mkdir(exist_ok=True)


def load_weights(name):
    return json.load(open(WEIGHTS / f"{name}.json"))


def done_keys(path, key):
    if not os.path.exists(path):
        return set()
    return {key(json.loads(l)) for l in open(path)}


def append(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
