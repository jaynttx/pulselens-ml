"""Optional local model cache. Network is needed once; no patient data is sent."""
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = 'e7bdd94cbe686611eb37f817bbdf48b8d770b057'
BASE = f'https://raw.githubusercontent.com/xan-antx/pulseLens/{REVISION}/apps/pulselens-deep/'
ROOT = Path(__file__).resolve().parents[1] / 'models'

def main():
    ROOT.mkdir(exist_ok=True)
    manifest = {'repository': 'xan-antx/pulseLens', 'revision': REVISION, 'files': {}}
    for name in ['model.tflite', 'state.gz']:
        with urlopen(BASE + 'models/' + name, timeout=90) as response:
            content = response.read()
        if name.endswith('.tflite') and content[4:8] != b'TFL3':
            raise ValueError('Downloaded asset is not a TFLite model')
        if name.endswith('.gz'):
            json.loads(gzip.decompress(content))
        (ROOT / name).write_bytes(content)
        manifest['files'][name] = {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        print(f'Cached {name}: {len(content):,} bytes')
    (ROOT / 'download-manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Model assets cached. Runtime and detector still need CDN access.')

if __name__ == '__main__':
    main()
