"""Python numerical and adapter tests; synthetic data is never a human benchmark."""
import unittest
import numpy as np
from pulselens.signals import estimate, pos, resample, detrend, bandpass
from pulselens.vitals import breathing, variability
from pulselens.tracking import FaceTracker, FrameClock
from pulselens.vision import resize_crop, SkinSampler, TemporalAmplifier
from pulselens.facephys import Model, FacePhys, OUTPUT_FOR_INPUT
from pulselens.session import Session
from pulselens.web import decode_frame


class SignalTests(unittest.TestCase):
    def test_known_rates(self):
        t = np.arange(600) / 30
        for bpm in [48, 60, 72, 90, 120, 150, 180]:
            with self.subTest(bpm=bpm):
                result = estimate(np.sin(2 * np.pi * bpm / 60 * t))
                self.assertTrue(result['valid']); self.assertAlmostEqual(result['bpm'], bpm, delta=1)

    def test_pos_with_illumination(self):
        t = np.arange(600) / 30; pulse = np.sin(2 * np.pi * 1.2 * t)
        light = 1 + .08 * np.sin(2 * np.pi * .1 * t)
        rgb = np.column_stack([130 * light + .1 * pulse, 100 * light + .8 * pulse, 75 * light - .1 * pulse])
        result = estimate(pos(rgb))
        self.assertTrue(result['valid']); self.assertAlmostEqual(result['bpm'], 72, delta=1)

    def test_drift_removed(self):
        t = np.arange(600) / 30
        result = estimate(np.sin(2 * np.pi * 1.2 * t) + 20 * np.sin(2 * np.pi * .12 * t))
        self.assertTrue(result['valid']); self.assertAlmostEqual(result['bpm'], 72, delta=1)

    def test_no_fake_rate_for_bad_input(self):
        for pulse in [np.zeros(600), np.full(600, np.nan), np.ones(20)]:
            self.assertIsNone(estimate(pulse))
        result = estimate(np.random.default_rng(42).normal(size=600))
        self.assertFalse(result['valid'])

    def test_resampling_and_timestamp_validation(self):
        result = resample([(0, [0]), (100, [10])], 20)
        np.testing.assert_allclose(result[:, 0], [0, 5, 10])
        with self.assertRaises(ValueError):
            resample([(1, [0]), (1, [10])])

    def test_detrend_and_filter(self):
        np.testing.assert_allclose(detrend(np.arange(100) * 3 + 10), 0, atol=1e-9)
        self.assertTrue(np.isfinite(bandpass(np.ones(100))).all())


class VitalTests(unittest.TestCase):
    def test_respiratory_rates(self):
        for rate in [12, 18, 24]:
            t = np.arange(1801) / 30
            rows = [(i * 1000 / 30, [130, 100 + np.sin(2 * np.pi * rate / 60 * t[i]), 75]) for i in range(len(t))]
            result = breathing(rows)
            self.assertTrue(result['valid']); self.assertAlmostEqual(result['rate'], rate, delta=.5)

    def test_short_and_flat_breathing(self):
        self.assertIsNone(breathing([(0, [1, 1, 1]), (1000, [1, 1, 1])]))
        self.assertIsNone(breathing([(0, [1, 1, 1]), (60000, [1, 1, 1])]))

    def test_regular_pulse_variability(self):
        t = np.arange(1800) / 30
        result = variability(np.sin(2 * np.pi * 1.2 * t), expected_bpm=72)
        self.assertTrue(result['valid']); self.assertLess(result['rmssd'], 2)
        self.assertIsNone(variability(np.ones(100), expected_bpm=72))
        self.assertIsNone(variability(np.zeros(1800), expected_bpm=72))


class ModelTests(unittest.TestCase):
    def test_recurrent_state_mapping(self):
        class Runner:
            def run(self, values):
                return [np.array([i], np.float32) for i in range(48)]
        model = FacePhys.__new__(FacePhys); model.model = Runner(); model.frames = 0
        model.values = [None] * 48
        pulse, visualization = model.step(np.full((36, 36, 3), 128), 1 / 30)
        self.assertEqual(pulse, 0); self.assertIsNone(visualization)
        for index in range(2, 48):
            self.assertEqual(model.values[index][0], OUTPUT_FOR_INPUT[index])
        self.assertAlmostEqual(float(model.values[1].flat[0]), 128 / 255, places=6)
        with self.assertRaises(ValueError):
            model.step(np.zeros((36, 36, 3)), float('nan'))

    def test_learned_psd_uses_actual_frame_rate(self):
        class Quality:
            def run(self, values):
                return [np.array([.8])]
        class Spectrum:
            def run(self, values):
                return [np.array([72]), np.array([0, 1.2, 3.5]), np.array([0, 1, 0]), np.array([1])]
        model = FacePhys.__new__(FacePhys); model.sqi = Quality(); model.psd = Spectrum()
        result = model.analyze(np.ones(450), 1 / 20)
        self.assertAlmostEqual(result['bpm'], 48); self.assertTrue(result['valid'])
        with self.assertRaises(ValueError):
            model.analyze(np.ones(449), 1 / 30)

    def test_python_tensor_order_is_normalized_to_model_contract(self):
        class Runtime:
            def get_input_details(self):
                return [dict(name='z', index=7, shape=[1], dtype=np.float32), dict(name='a', index=2, shape=[1], dtype=np.float32)]
            def get_output_details(self):
                return [dict(name='z', index=8), dict(name='a', index=3)]
            def set_tensor(self, index, value):
                self.bound = getattr(self, 'bound', {})
                self.bound[index] = value.copy()
            def invoke(self):
                pass
            def get_tensor(self, index):
                return np.array([index])
        runtime = Runtime(); model = Model(runtime)
        outputs = model.run([np.array([11]), np.array([22])])
        self.assertEqual(runtime.bound[2][0], 11)
        self.assertEqual(runtime.bound[7][0], 22)
        self.assertEqual([value[0] for value in outputs], [3, 8])

    def test_runtime_tensor_shape_checks(self):
        class Runtime:
            def get_input_details(self):
                return [dict(name='input', index=7, shape=[1, 2], dtype=np.float32)]
            def get_output_details(self):
                return [dict(name='output', index=8, shape=[1, 2])]
            def set_tensor(self, index, value):
                self.value = value
            def invoke(self):
                pass
            def get_tensor(self, index):
                return self.value * 2
        model = Model(Runtime())
        np.testing.assert_allclose(model.run([np.array([[1, 2]])])[0], [[2, 4]])
        with self.assertRaises(ValueError):
            model.run([np.zeros((2, 1))])


class TrackingAndPixelsTests(unittest.TestCase):
    def test_kalman(self):
        tracker = FaceTracker(); tracker.update([100, 80, 160, 180])
        result = tracker.update([110, 80, 160, 180])
        self.assertAlmostEqual(result[0], 100 + 10 * 1.01 / 1.51)

    def test_clock(self):
        clock = FrameClock(); previous = -1
        for i in range(1000):
            dt, now = clock.update(1000 + i * 1000 / 30 + (3 if i % 2 else 0))
            self.assertGreater(now, previous); previous = now
            self.assertLess(abs(dt - 1 / 30), .00002)
        with self.assertRaises(ValueError):
            clock.update(1)

    def test_crop_and_spatial_sampling(self):
        image = np.zeros((100, 100, 3), np.uint8); image[:] = [130, 100, 75]
        crop = resize_crop(image, [-10, -10, 80, 80], 36)
        self.assertEqual(crop.shape, (36, 36, 3)); np.testing.assert_allclose(crop[0, 0], [130, 100, 75])
        sampler = SkinSampler()
        rgb = sampler.sample(image, [[0, 0, 30, 30]] * 3, None, 1000)
        np.testing.assert_allclose(rgb, [130, 100, 75])
        self.assertIsNone(SkinSampler().sample(np.zeros_like(image), [[0, 0, 30, 30]] * 3, None, 1000))

    def test_amplification_preserves_constant_pixels(self):
        amplifier = TemporalAmplifier(); image = np.full((36, 36, 3), 120)
        for _ in range(30):
            self.assertEqual(amplifier.process(image, 1 / 30, 10), [120] * (36 * 36 * 3))

    def test_raw_rgba_input_keeps_camera_values(self):
        rgba = np.tile(np.array([130, 100, 75, 255], np.uint8), 36 * 36)
        result = decode_frame(rgba.tobytes(), 36, 36)
        np.testing.assert_array_equal(result[0, 0], [130, 100, 75])
        with self.assertRaises(ValueError):
            decode_frame(b'not an image', 640, 480)


class FakeVision:
    def __init__(self, method):
        self.closed = False
    def process(self, image, now, bpm):
        pulse = np.sin(2 * np.pi * 1.2 * now / 1000)
        rgb = np.array([130 + .1 * pulse, 100 + .8 * pulse, 75 - .1 * pulse])
        return dict(box=dict(x=0, y=0, w=36, h=36), crop=np.full((36, 36, 3), 128), rgb=rgb, thumbnail=[128] * 3888)
    def close(self):
        self.closed = True


class SessionTests(unittest.TestCase):
    def test_python_capture_to_pos_to_bpm(self):
        session = Session('pos', vision_factory=FakeVision)
        results = []
        for i in range(650):
            output = session.process(np.zeros((36, 36, 3), np.uint8), i * 1000 / 30)
            if output.get('result') and output['result']['valid']:
                results.append(output['result']['bpm'])
        self.assertTrue(results); self.assertAlmostEqual(results[-1], 72, delta=1)
        self.assertGreater(output['capture']['pulseSeconds'], 20)
        vision = session.vision; session.close(); self.assertTrue(vision.closed)

    def test_timestamp_validation_and_buffer_limit(self):
        session = Session('pos', vision_factory=FakeVision)
        session.process(np.zeros((36, 36, 3), np.uint8), 1000)
        with self.assertRaises(ValueError):
            session.process(np.zeros((36, 36, 3), np.uint8), 999)
        session.close()

    def test_research_breathing_does_not_need_a_bpm_result(self):
        session = Session('pos', vision_factory=FakeVision)
        for i in range(1801):
            now = i * 1000 / 30
            session.rows.append((now, [100, 100, 100]))
            session.rgb_rows.append((now, [130, 100 + np.sin(2 * np.pi * .3 * i / 30), 75]))
        result = session.analyze()
        self.assertIsNone(result['result']); self.assertTrue(result['insights']['breathing']['valid'])
        session.close()

if __name__ == '__main__':
    unittest.main()

class CpuFaceAdapterTests(unittest.TestCase):
    def test_blazeface_decoding_on_cpu(self):
        from pulselens.face_models import FaceModels
        class Detector:
            outputs = [dict(name='classificators'), dict(name='regressors')]
            def run(self, values):
                scores = np.full((1, 896, 1), -100., np.float32); scores[0, 272, 0] = 20
                boxes = np.zeros((1, 896, 16), np.float32); boxes[0, 272, 2:4] = 64
                return [scores, boxes]
        model = FaceModels.__new__(FaceModels); model.detector = Detector()
        anchors = [(x + .5, y + .5) for grid, repeats in [(16, 2), (8, 6)] for y in range(grid) for x in range(grid) for _ in range(repeats)]
        model.anchors = np.array(anchors, dtype=float); model.anchors[:512] /= 16; model.anchors[512:] /= 8
        result = model.detect(np.zeros((128, 128, 3), np.uint8))
        self.assertEqual(len(result), 1); np.testing.assert_allclose(result[0]['box'], [36, 36, 64, 64], atol=1e-5)

    def test_face_preprocessing_padding_and_confidence(self):
        from pulselens.face_models import sample_grid, sigmoid
        image = np.full((4, 4, 3), 100, np.uint8)
        result = sample_grid(image, np.array([[-2., 1.5]]), np.array([[1., 1.5]]))
        np.testing.assert_allclose(result[0, 0], [0, 0, 0]); np.testing.assert_allclose(result[0, 1], [100, 100, 100])
        np.testing.assert_allclose(sigmoid(np.array([0.])), [.5])

class LiveCaptureRegressionTests(unittest.TestCase):
    def test_live_pos_rates_with_irregular_frames_and_lighting(self):
        from pulselens.classical import estimate_live_rgb
        for bpm in [48, 60, 77, 120, 150, 180]:
            with self.subTest(bpm=bpm):
                t = np.cumsum(np.tile([.025, .045, .033, .08], 180))
                pulse = np.sin(2 * np.pi * bpm / 60 * t)
                light = 1 + .12 * np.sin(2 * np.pi * .15 * t)
                rgb = np.column_stack([130 * light + .1 * pulse, 100 * light + .8 * pulse, 75 * light - .1 * pulse])
                rows = [(x * 1000, value.tolist()) for x, value in zip(t, rgb)]
                result = estimate_live_rgb(rows)
                self.assertTrue(result['valid'])
                self.assertAlmostEqual(result['bpm'], bpm, delta=1)
                self.assertGreater(result['windowSeconds'], 19)
                self.assertLessEqual(result['windowSeconds'], 20)

    def test_live_pos_does_not_invent_a_rate_for_missing_or_flat_signal(self):
        from pulselens.classical import estimate_live_rgb
        self.assertIsNone(estimate_live_rgb([(0, [130, 100, 75]), (20000, [130, 100, 75])]))
        self.assertIsNone(estimate_live_rgb([(i * 1000 / 30, [130, 100, 75]) for i in range(600)]))
        t = np.arange(200) / 30
        self.assertIsNone(estimate_live_rgb([(x * 1000, [130, 100 + np.sin(x), 75]) for x in t]))

    def test_respiratory_preview_before_rate_and_serializable_trace(self):
        import json
        t = np.arange(1801) / 30
        rows = [(x * 1000, [130, 100 + np.sin(2 * np.pi * .3 * x), 75]) for x in t]
        preview = breathing(rows[:451])
        self.assertFalse(preview['valid']); self.assertIsNone(preview['rate'])
        self.assertGreater(len(preview['waveform']), 100)
        result = breathing(rows)
        self.assertTrue(result['valid']); self.assertAlmostEqual(result['rate'], 18, delta=.5)
        self.assertLessEqual(len(result['waveform']), 601)
        json.dumps(result, allow_nan=False)
