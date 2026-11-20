"""Launch the local Phase 19 Face24 QR Gradio application."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

main = importlib.import_module("demo.codec").main


if __name__ == "__main__":
    main()
