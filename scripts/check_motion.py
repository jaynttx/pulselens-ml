"""Measure frame processing and retention using a public *static* test portrait.

This is a controlled motion/occlusion test, NOT a human pulse-accuracy test.
Optional dependency for JPEG decoding: python -m pip install Pillow.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulselens.session import Session


def run(image_path, frames=1900):
    from PIL import Image
    image = Image.open(image_path).convert('RGB'); image.thumbnail((640, 480)); base = np.asarray(image)
    report = dict(scope='Controlled motion on a public static portrait; not live heart-rate validation.', methods={})
    for method in ['pos', 'deep']:
        session = Session(method); latency = []; detected = analyses = 0; started = time.perf_counter()
        try:
            for index in range(frames):
                frame = np.roll(base, (int(2 * np.sin(index / 17)), int(3 * np.sin(index / 13))), axis=(0, 1))
                if 600 <= index < 620:
                    frame = np.full_like(base, 128)
                tick = time.perf_counter()
                output = session.process(frame, index * 1000 / 30, amplify=index % 5 == 0)
                latency.append((time.perf_counter() - tick) * 1000)
                detected += bool(output['face']); analyses += 'result' in output
            summary = dict(frames=frames, detected_frames=detected, intentional_occluded_frames=20,
                retained_samples=len(session.rows), analysis_updates=analyses,
                pulse_window_seconds=session.capture()['pulseSeconds'], color_window_seconds=session.capture()['colorSeconds'],
                median_processing_ms=round(float(np.median(latency)), 3), p95_processing_ms=round(float(np.percentile(latency, 95)), 3),
                elapsed_seconds=round(time.perf_counter() - started, 2))
            if detected < frames - 25 or summary['color_window_seconds'] < 59:
                raise AssertionError(f'Face tracking or signal history failed: {summary}')
            report['methods'][method] = summary
        finally:
            session.close()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run(args.image)
    content = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(content)
    print(content)
