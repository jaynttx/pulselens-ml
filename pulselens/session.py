"""One user's in-memory measurement session. No image/metric files are written."""
from collections import deque
from threading import Lock
import time
import numpy as np
from .signals import resample, pos, estimate, bandpass
from .vitals import breathing, variability
from .tracking import FrameClock
from .vision import FaceVision, TemporalAmplifier
from .facephys import FacePhys


class Session:
    def __init__(self, method, vision_factory=FaceVision, model_factory=FacePhys):
        if method not in ('pos', 'deep'):
            raise ValueError('Unknown measurement method.')
        self.method = method; self.lock = Lock(); self.last_used = time.monotonic()
        self.model = None; self.vision = None
        try:
            self.model = model_factory() if method == 'deep' else None
            self.vision = vision_factory(method)
        except Exception:
            if self.vision:
                self.vision.close()
            raise
        self.clock = FrameClock(); self.rows = deque(); self.rgb_rows = deque()
        self.last_analysis = -10000.; self.last_frame = None; self.last_good_bpm = None
        self.last_face = None; self.frames = 0; self.amplifier = TemporalAmplifier()

    def clear_window(self):
        self.rows.clear(); self.rgb_rows.clear(); self.clock = FrameClock()
        self.last_good_bpm = None; self.last_analysis = -10000
        self.amplifier = TemporalAmplifier()
        if self.model:
            self.model.reset()

    @staticmethod
    def duration(rows):
        return max(0, (rows[-1][0] - rows[0][0]) / 1000) if len(rows) > 1 else 0

    def process(self, image, now, amplify=False, gain=10):
        if not np.isfinite(now) or (self.last_frame is not None and now <= self.last_frame):
            raise ValueError('Frame timestamps must strictly increase.')
        started = time.perf_counter(); self.last_used = time.monotonic()
        dt = (now - self.last_frame) / 1000 if self.last_frame is not None else 1 / 30
        self.last_frame = now
        if dt > .75:
            self.clear_window()
        face = self.vision.process(image, now, self.last_good_bpm)
        if face is None:
            if self.last_face is not None and now - self.last_face > 1500:
                self.clear_window(); self.last_face = None
            return dict(face=False, guidance='Face not detected. Keep one face in view.', capture=self.capture())
        self.last_face = now; self.frames += 1
        self.rgb_rows.append((now, face['rgb'].tolist()))
        while self.rgb_rows and now - self.rgb_rows[0][0] > 60000:
            self.rgb_rows.popleft()
        output = dict(face=True, box=face['box'], capture=self.capture(), sample=None)
        if self.method == 'pos':
            if min(face['rgb']) < 1 or face['rgb'].mean() < 18 or face['rgb'].mean() > 245:
                return dict(face=False, guidance='Use even lighting.', capture=self.capture())
            stamp, value = now, face['rgb'].tolist()
        else:
            interval, stamp = self.clock.update(now)
            pulse, visualization = self.model.step(face['crop'], interval)
            value = [pulse]
            if visualization:
                output['visualization'] = visualization
        self.rows.append((stamp, value))
        while self.rows and stamp - self.rows[0][0] > 60000:
            self.rows.popleft()
        output['sample'] = dict(t=stamp, v=value)
        output['capture'] = self.capture()
        if self.frames % 5 == 0:
            output['crop'] = face['thumbnail']
            if amplify:
                output['amplified'] = self.amplifier.process(face['crop'], dt, gain)
        minimum = 450 if self.method == 'deep' else 150
        if len(self.rows) >= minimum and now - self.last_analysis >= 1000:
            self.last_analysis = now
            output.update(self.analyze())
        output['latency'] = (time.perf_counter() - started) * 1000
        return output

    def capture(self):
        return dict(pulseSeconds=self.duration(self.rows), colorSeconds=self.duration(self.rgb_rows), samples=len(self.rows))

    def analyze(self):
        rows = list(self.rows); rgb_rows = list(self.rgb_rows)
        recent = [row for row in rows if rows[-1][0] - row[0] <= 20000]
        uniform = resample(recent)
        pulse = pos(uniform) if self.method == 'pos' else uniform[:, 0]
        result = estimate(pulse)
        if self.method == 'deep':
            raw = rows[-450:]
            interval = (raw[-1][0] - raw[0][0]) / 449000
            learned = self.model.analyze([row[1][0] for row in raw], interval)
            if result is None:
                wave = bandpass(np.array([row[1][0] for row in raw]))
                result = dict(waveform=(wave / max(wave.std(), 1e-12)).tolist())
            result.update(learned)
        if result and result.get('valid'):
            self.last_good_bpm = result['bpm']
        whole = resample(rows)
        long_pulse = pos(whole) if self.method == 'pos' else whole[:, 0]
        long_estimate = estimate(long_pulse) if len(long_pulse) >= 30 * 55 else None
        expected = long_estimate['bpm'] if long_estimate and long_estimate['valid'] else self.last_good_bpm
        insights = dict(capture=self.capture(), breathing=breathing(rgb_rows), variability=variability(long_pulse, expected_bpm=expected))
        if result:
            result.update(breathing=insights['breathing'], variability=insights['variability'])
        return dict(result=result, insights=insights)

    def close(self):
        if self.vision:
            self.vision.close()
        self.rows.clear(); self.rgb_rows.clear(); self.model = None
