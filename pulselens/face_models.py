"""CPU adapters for MediaPipe's pretrained BlazeFace and face-landmark weights.

These reproduce the model preprocessing/decoding contracts in Google's face
pipeline, without its native GPU graph (which aborts on some headless Macs).
No face model is trained here. See THIRD_PARTY_NOTICES.md.
"""
import zipfile
import numpy as np
from .assets import ensure
from .facephys import Model, interpreter


def sample_grid(image, xs, ys):
    height, width = image.shape[:2]
    inside = (xs >= -.5) & (xs < width - .5) & (ys >= -.5) & (ys < height - .5)
    xs, ys = np.clip(xs, 0, width - 1), np.clip(ys, 0, height - 1)
    left, top = np.floor(xs).astype(int), np.floor(ys).astype(int)
    right, bottom = np.minimum(left + 1, width - 1), np.minimum(top + 1, height - 1)
    ax, ay = (xs - left)[..., None], (ys - top)[..., None]
    result = ((image[top, left] * (1 - ax) + image[top, right] * ax) * (1 - ay) +
              (image[bottom, left] * (1 - ax) + image[bottom, right] * ax) * ay)
    return result * inside[..., None]


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -80, 80)))


class FaceModels:
    def __init__(self, landmarks=False):
        self.detector = Model(interpreter(ensure('blaze_face_short_range.tflite')))
        self.landmarker = None
        if landmarks:
            with zipfile.ZipFile(ensure('face_landmarker.task')) as bundle:
                self.landmarker = Model(interpreter(bundle.read('face_landmarks_detector.tflite')))
        # Short-range BlazeFace: stride-8 grid (two anchors/location), then
        # three merged stride-16 layers (six anchors/location), fixed sizes.
        self.anchors = np.array([(x + .5, y + .5) for grid, repeats in [(16, 2), (8, 6)]
                                 for y in range(grid) for x in range(grid) for _ in range(repeats)], dtype=float)
        self.anchors[:512] /= 16; self.anchors[512:] /= 8

    def detect(self, image):
        height, width = image.shape[:2]
        scale = 128 / max(width, height)
        padding = np.array([(128 - width * scale) / 2, (128 - height * scale) / 2])
        grid = np.arange(128) + .5
        xs, ys = np.meshgrid((grid - padding[0]) / scale - .5, (grid - padding[1]) / scale - .5)
        pixels = sample_grid(image, xs, ys).astype(np.float32) / 127.5 - 1
        output = self.detector.run([pixels[None]])
        by_name = dict(zip([spec['name'] for spec in self.detector.outputs], output))
        scores = sigmoid(by_name['classificators'].reshape(-1))
        raw = by_name['regressors'].reshape(896, 16)
        selected = np.flatnonzero(scores >= .5)
        if not len(selected):
            return []
        centers = raw[selected, :2] / 128 + self.anchors[selected]
        sizes = raw[selected, 2:4] / 128
        boxes = np.column_stack([centers - sizes / 2, centers + sizes / 2])
        points = raw[selected, 4:].reshape(-1, 6, 2) / 128 + self.anchors[selected, None, :]
        remaining = list(np.argsort(scores[selected])[::-1]); detections = []
        while remaining:
            best = remaining[0]; box = boxes[best]
            all_boxes = boxes[remaining]
            overlap = np.maximum(0, np.minimum(box[2:], all_boxes[:, 2:]) - np.maximum(box[:2], all_boxes[:, :2]))
            intersection = overlap.prod(axis=1)
            area = np.maximum(0, all_boxes[:, 2:] - all_boxes[:, :2]).prod(axis=1)
            own_area = np.maximum(0, box[2:] - box[:2]).prod()
            matched = intersection / np.maximum(area + own_area - intersection, 1e-9) > .3
            group = np.array(remaining)[matched]; weights = scores[selected[group]]
            merged = np.average(boxes[group], axis=0, weights=weights)
            keypoints = np.average(points[group], axis=0, weights=weights)
            lo = (merged[:2] * 128 - padding) / scale
            hi = (merged[2:] * 128 - padding) / scale
            keypoints = (keypoints * 128 - padding) / scale
            detections.append(dict(box=np.r_[lo, hi - lo], keypoints=keypoints))
            remaining = np.array(remaining)[~matched].tolist()
        return detections

    def landmarks(self, image, detection):
        if self.landmarker is None:
            raise ValueError('Landmark model is not loaded.')
        x, y, width, height = detection['box']
        center = np.array([x + width / 2, y + height / 2]); size = max(width, height) * 1.5
        eye_vector = detection['keypoints'][1] - detection['keypoints'][0]
        angle = np.arctan2(eye_vector[1], eye_vector[0])
        if angle > np.pi / 2:
            angle -= np.pi
        if angle < -np.pi / 2:
            angle += np.pi
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        axis = (np.arange(256) + .5) / 256 - .5
        xx, yy = np.meshgrid(axis, axis)
        grid = np.stack([xx, yy], axis=-1) * size @ rotation.T + center
        pixels = sample_grid(image, grid[:, :, 0] - .5, grid[:, :, 1] - .5).astype(np.float32) / 255
        outputs = self.landmarker.run([pixels[None]])
        mapped = dict(zip([spec['name'] for spec in self.landmarker.outputs], outputs))
        if float(sigmoid(mapped['Identity_1'].flat[0])) < .5:
            return None
        points = mapped['Identity'].reshape(478, 3)
        return (points[:, :2] / 256 - .5) * size @ rotation.T + center

    def close(self):
        self.detector = self.landmarker = None
