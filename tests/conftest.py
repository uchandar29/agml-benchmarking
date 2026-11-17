import os
import sys
from pathlib import Path

# faiss and torch each bundle libomp, which crashes on macOS. Harmless on Linux.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Lets a plain `pytest` run from the repo root import core and benchmark
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
