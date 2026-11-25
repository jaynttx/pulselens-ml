"""Python MediaPipe detection, skin sampling and temporal pixel amplification."""
from collections import deque
import numpy as np
from .assets import ensure
from .tracking import FaceTracker

PATCHES = [[10, 67, 69, 104, 108, 151, 284, 298, 299, 337],
           [123, 147, 187, 205, 206, 216], [352, 376, 411, 425, 426, 436]]


def resize_crop(image, box, size):
    height, width = image.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Face crop lies outside the image.')
    xs = np.clip(x0 + (np.arange(size) + .5) * (x1 - x0) / size - .5, 0, width - 1)
    ys = np.clip(y0 + (np.arange(size) + .5) * (y1 - y0) / size - .5, 0, height - 1)
    left, top = np.floor(xs).astype(int), np.floor(ys).astype(int)
    right, bottom = np.minimum(left + 1, width - 1), np.minimum(top + 1, height - 1)
    ax, ay = (xs - left)[None, :, None], (ys - top)[:, None, None]
    upper = image[top[:, None], left] * (1 - ax) + image[top[:, None], right] * ax
    lower = image[bottom[:, None], left] * (1 - ax) + image[bottom[:, None], right] * ax
    return upper * (1 - ay) + lower * ay


class SkinSampler:
    def __init__(self):
        self.history = [deque(maxlen=64) for _ in range(3)]
        self.weights = [np.ones(256) for _ in range(3)]
        self.frames = 0

    def sample(self, image, patches, bpm, now):
        total = np.zeros(3); mass = 0.
        for index, box in enumerate(patches):
            rgb = resize_crop(image, box, 16).reshape(-1, 3)
            sums = np.maximum(rgb.sum(axis=1), 1)
            rn, gn = rgb[:, 0] / sums, rgb[:, 1] / sums
            mask = (sums >= 60) & (rn > .35) & (rn < .6) & (gn > .25) & (gn < .4)
            weights = self.weights[index] * mask
            total += (rgb * weights[:, None]).sum(axis=0); mass += weights.sum()
            history = self.history[index]; history.append((now, gn.copy()))
            if bpm and len(history) == 64 and self.frames % 15 == 0:
                times = np.array([row[0] for row in history])
                signal = np.stack([row[1] for row in history]); signal -= signal.mean(axis=0)
                phase = 2 * np.pi * bpm / 60 * (times - times[0]) / 1000
                power = (np.cos(phase) @ signal) ** 2 + (np.sin(phase) @ signal) ** 2
                coherence = np.minimum(1, 2 * power / np.maximum(1e-12, 64 * (signal ** 2).sum(axis=0)))
                self.weights[index] = .1 + .9 * coherence
        self.frames += 1
        return total / mass if mass else None


class FaceVision:
    def __init__(self, method):
        from .face_models import FaceModels
        self.method = method
        self.detector = FaceModels(landmarks=method == 'pos')
        self.tracker = FaceTracker(); self.skin = SkinSampler(); self.last_timestamp = -1

    def process(self, image, now, bpm=None):
        stamp = max(self.last_timestamp + 1, round(now)); self.last_timestamp = stamp
        detections = self.detector.detect(image)
        patches = None
        if len(detections) != 1:
            return None
        raw = detections[0]['box']
        if self.method == 'pos':
            points = self.detector.landmarks(image, detections[0])
            if points is None:
                return None
            def bounds(values, padding=0):
                lo, hi = values.min(axis=0) - padding, values.max(axis=0) + padding
                return [*lo, *(hi - lo)]
            raw = bounds(points)
            patches = [bounds(points[ids], 4) for ids in PATCHES]
        box = self.tracker.update(raw)
        x, y, w, h = box
        face = resize_crop(image, [x, y - h * .24, w, h * 1.2], 36)
        rgb = self.skin.sample(image, patches, bpm, now) if patches else resize_crop(image, box, 32).mean(axis=(0, 1))
        if rgb is None:
            return None
        return dict(box=dict(zip(('x', 'y', 'w', 'h'), box.tolist())), rgb=rgb, crop=face,
                    thumbnail=np.clip(face, 0, 255).astype(np.uint8).ravel().tolist())

    def close(self):
        self.detector.close()


class TemporalAmplifier:
    def __init__(self):
        self.fast = self.slow = None

    def process(self, pixels, dt, gain):
        pixels = np.asarray(pixels, dtype=float)
        if self.fast is None:
            self.fast = pixels.copy(); self.slow = pixels.copy()
        a = 1 - np.exp(-2 * np.pi * 3.33 * min(dt, .1))
        b = 1 - np.exp(-2 * np.pi * .67 * min(dt, .1))
        self.fast += a * (pixels - self.fast); self.slow += b * (pixels - self.slow)
        return np.clip(pixels + np.clip(gain, 1, 30) * (self.fast - self.slow), 0, 255).astype(np.uint8).ravel().tolist()
