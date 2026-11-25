"""Execute real pretrained Python models with artificial test input, no camera."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulselens.web import model_check

if __name__ == '__main__':
    print(json.dumps(model_check(), indent=2))
