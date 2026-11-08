from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_research.pipeline import run_pipeline


if __name__ == "__main__":
    summary = run_pipeline(PROJECT_ROOT / "configs" / "base.json")
    print("\nPipeline complete")
    print(f"Outputs: {summary['outputs']}")
    print(f"Reports: {summary['reports']}")

