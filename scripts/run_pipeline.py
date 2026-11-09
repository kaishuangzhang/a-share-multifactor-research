from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant_research.pipeline import run_pipeline


if __name__ == "__main__":
    summary = run_pipeline(PROJECT_ROOT / "configs" / "base.json")
    print("\n项目流程执行完成")
    print(f"数值结果目录：{summary['outputs']}")
    print(f"报告目录：{summary['reports']}")

