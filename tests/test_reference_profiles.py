"""Regression checks for processing settings verified against the reference."""
import unittest
import numpy as np
from pulselens.classical import estimate_rgb, extract_pos, local_detrend
from pulselens.session import Session
from pulselens.vision import FaceVision, SkinSampler, resize_crop
from pulselens.tracking import FaceTracker
from test_python_pipeline import FakeVision


class ReferenceProfileTests(unittest.TestCase):
    def test_classical_calibration_and_reference_window(self):
        rows = [(i * 1000 / 30, [130 + .1 * np.sin(i / 4), 100 + .8 * np.sin(i / 4), 75 - .1 * np.sin(i / 4)]) for i in range(400)]
        self.assertIsNone(estimate_rgb(rows[:149]))
        full, tail = estimate_rgb(rows), estimate_rgb(rows[-256:])
        self.assertAlmostEqual(full['bpm'], tail['bpm'], places=12)
        self.assertEqual(len(full['waveform']), 256)

    def test_reference_known_classical_rates(self):
        for rate in [48, 60, 72, 90, 120, 150, 180]:
            with self.subTest(rate=rate):
                t = np.arange(256) / 30; pulse = np.sin(2 * np.pi * rate / 60 * t)
                rgb = np.column_stack([130 + .1 * pulse, 100 + .8 * pulse, 75 - .1 * pulse])
                result = estimate_rgb([(i * 1000 / 30, value.tolist()) for i, value in enumerate(rgb)])
                self.assertTrue(result['valid']); self.assertAlmostEqual(result['bpm'], rate, delta=2)

    def test_neural_psd_uses_current_smoothed_interval(self):
        class Model:
            def analyze(self, samples, dt):
                self.received = (samples, dt)
                return dict(bpm=72., frequency=1.2, quality=.8, valid=True, learnedSqi=.8, powers=[], frequencies=[])
            def reset(self):
                pass
        model = Model()
        session = Session('deep', vision_factory=FakeVision, model_factory=lambda: model)
        for i in range(480):
            session.rows.append((i * 1000 / 30, [float(i)]))
        session.clock.dt = .05
        session.analyze()
        self.assertEqual(model.received[1], .05)
        self.assertEqual(model.received[0], list(map(float, range(30, 480))))
        session.close()

    def test_short_capture_stall_does_not_erase_history(self):
        session = Session('pos', vision_factory=FakeVision)
        image = np.zeros((36, 36, 3), np.uint8)
        for now in [0, 33, 933]:
            session.process(image, now)
        self.assertEqual(len(session.rows), 3)
        session.process(image, 4000)
        self.assertEqual(len(session.rows), 1)
        session.close()

    def test_crop_is_8bit_like_a_canvas_readback(self):
        class Detector:
            def detect(self, image):
                return [dict(box=np.array([5.3, 10.2, 40.5, 45.4]))]
        vision = FaceVision.__new__(FaceVision)
        vision.method = 'deep'; vision.detector = Detector(); vision.tracker = FaceTracker(); vision.skin = SkinSampler(); vision.last_timestamp = -1
        image = np.arange(80 * 80 * 3, dtype=np.uint8).reshape(80, 80, 3)
        result = vision.process(image, 1000)
        self.assertEqual(result['crop'].dtype, np.uint8)
        expected = np.rint(np.clip(resize_crop(image, [5.3, 10.2 - 45.4 * .24, 40.5, 45.4 * 1.2], 36), 0, 255)).astype(np.uint8)
        np.testing.assert_array_equal(result['crop'], expected)

    def test_local_mean_removal_handles_edges(self):
        values = np.arange(100, dtype=float)
        expected = np.array([values[max(0, i - 30):min(100, i + 30)].mean() for i in range(100)])
        np.testing.assert_allclose(local_detrend(values), values - expected)
        np.testing.assert_array_equal(extract_pos(np.full((150, 3), 100)), np.zeros(150))
