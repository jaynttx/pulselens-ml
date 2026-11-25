"""Scalar Kalman face tracking and recurrent-model frame clock."""
import numpy as np


class FaceTracker:
    def __init__(self):
        self.values = None
        self.variance = np.ones(4)

    def update(self, box):
        box = np.asarray(box, dtype=float)
        if self.values is None:
            self.values = box.copy()
        else:
            predicted = self.variance + .01
            gain = predicted / (predicted + .5)
            self.values += gain * (box - self.values)
            self.variance = (1 - gain) * predicted
        return self.values.copy()


class FrameClock:
    def __init__(self):
        self.last = None; self.time = None; self.dt = 1 / 30

    def update(self, now):
        if not np.isfinite(now) or (self.last is not None and now <= self.last):
            raise ValueError('Frame timestamps must increase.')
        if self.last is None:
            self.time = now
        else:
            self.dt += .003 * ((now - self.last) / 1000 - self.dt)
            self.time += self.dt * 1000
            self.time += .003 * (now - self.time)
        self.last = now
        return self.dt, self.time
