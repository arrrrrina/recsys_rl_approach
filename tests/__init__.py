"""Enable `python3 -m unittest discover -s tests -t .` without PYTHONPATH."""
from pathlib import Path
import sys

source = str(Path(__file__).resolve().parents[1] / 'src')
if source not in sys.path:
    sys.path.insert(0, source)
