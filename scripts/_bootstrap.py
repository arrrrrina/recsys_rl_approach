"""Allow direct script execution from an activated Python environment."""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bootstrap():
    source = str(PROJECT_ROOT / 'src')
    if source not in sys.path:
        sys.path.insert(0, source)
    # All data/config arguments are relative to the repository, including when
    # invoked as `python3 train_baseline.py` from the scripts directory.
    os.chdir(PROJECT_ROOT)
