"""Check the fixed Conda/PyTorch environment."""
from __future__ import annotations
import importlib.util
import os
import sys

EXPECTED = os.path.normcase(r"D:\conda_envs\python312\python.exe")

def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    if os.path.normcase(sys.executable) != EXPECTED:
        print(f"WARNING: expected interpreter: {EXPECTED}")
    for name in ("torch", "transformers", "datasets", "peft", "accelerate"):
        print(f"{name}: {'installed' if importlib.util.find_spec(name) else 'MISSING'}")
    try:
        import torch
        print(f"Torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GB")
    except ImportError:
        print("Torch is not installed. Install the CUDA wheel first.")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
