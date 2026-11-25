"""Download pinned model weights and MediaPipe detectors; no camera access."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulselens.assets import ensure, NEURAL, DETECTORS

if __name__ == '__main__':
    for name in (*NEURAL, *DETECTORS):
        path = ensure(name)
        print(f'Ready: {name} ({path.stat().st_size:,} bytes)')
