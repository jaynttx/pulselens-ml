"""Pinned pretrained assets. This module downloads models, never camera frames."""
import gzip
import json
import ssl
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
REVISION = 'e7bdd94cbe686611eb37f817bbdf48b8d770b057'
MODEL_URL = f'https://raw.githubusercontent.com/xan-antx/pulseLens/{REVISION}/apps/pulselens-deep/models/'
DETECTORS = {
    'blaze_face_short_range.tflite': 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
    'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
}
NEURAL = ('model.tflite', 'proj.tflite', 'sqi_model.tflite', 'psd_model.tflite', 'state.gz')


def ensure(name):
    path = MODEL_DIR / name
    if path.is_file():
        return path
    MODEL_DIR.mkdir(exist_ok=True)
    url = DETECTORS.get(name, MODEL_URL + name)
    try:
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urlopen(url, timeout=90, context=context) as response:
            content = response.read()
        if name.endswith('.tflite') and content[4:8] != b'TFL3':
            raise ValueError('Invalid TFLite asset: ' + name)
        if name.endswith('.gz'):
            json.loads(gzip.decompress(content))
        temporary = path.with_suffix(path.suffix + '.part')
        temporary.write_bytes(content); temporary.replace(path)
    except Exception as error:
        raise RuntimeError(f'Cannot download {name}. Check internet access, then run python scripts/download_models.py. {error}') from error
    return path
