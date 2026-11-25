"""Experimental respiratory proxy and pulse-rate variability, implemented in Python."""
import numpy as np
from .signals import detrend, resample


def slow_filter(values, fs=30, low=.08, high=.6):
    if not len(values):
        return np.array([])
    a, b = np.exp(-2 * np.pi * low / fs), 1 - np.exp(-2 * np.pi * high / fs)
    previous, hp, lp = values[0], 0., 0.
    output = []
    for value in values:
        hp = a * (hp + value - previous); previous = value
        lp += b * (hp - lp); output.append(lp)
    return np.asarray(output)


def breathing(rows, fs=30):
    if len(rows) < 2 or rows[-1][0] - rows[0][0] < 5000:
        return None
    green = resample(rows, fs)[:, 1]
    if green.std() < 1e-8:
        return None
    signal = slow_filter(detrend(green), fs)[fs * 3:]
    spread = signal.std()
    if spread < 1e-10:
        return None
    trace = dict(waveform=(signal / spread)[::3].tolist(), sampleRate=fs / 3,
                 seconds=len(signal) / fs)
    if rows[-1][0] - rows[0][0] < 30000:
        return dict(rate=None, quality=0., valid=False, **trace)
    rates = np.arange(6, 30.01, .25)
    phases = 2 * np.pi * rates[:, None] / 60 * np.arange(len(signal)) / fs
    tapered = signal * np.hanning(len(signal))
    powers = (np.cos(phases) @ tapered) ** 2 + (np.sin(phases) @ tapered) ** 2
    peak = int(powers.argmax()); rate = rates[peak]
    quality = float(powers[np.abs(rates - rate) <= 2].sum() / max(powers.sum(), 1e-12))
    return dict(rate=float(rate), quality=quality, valid=bool(quality > .55 and 0 < peak < len(rates) - 1), **trace)


def variability(pulse, fs=30, expected_bpm=None):
    if len(pulse) / fs < 55 or expected_bpm is None or not np.isfinite(expected_bpm):
        return None
    filtered = slow_filter(detrend(pulse), fs, .67, 3.33)
    skip = fs * 3; spread = filtered[skip:].std()
    if spread < 1e-10:
        return None
    refractory = max(.3, 60 / expected_bpm * .65) * fs
    peaks = []
    for i in range(skip + 1, len(filtered) - 1):
        if filtered[i] < spread * .15 or filtered[i] <= filtered[i - 1] or filtered[i] < filtered[i + 1]:
            continue
        curvature = filtered[i - 1] - 2 * filtered[i] + filtered[i + 1]
        offset = np.clip(.5 * (filtered[i - 1] - filtered[i + 1]) / curvature, -.5, .5) if curvature else 0
        point = (i + float(offset), filtered[i])
        if peaks and point[0] - peaks[-1][0] < refractory:
            if point[1] > peaks[-1][1]:
                peaks[-1] = point
        else:
            peaks.append(point)
    intervals = np.diff([point[0] for point in peaks]) * 1000 / fs
    if len(intervals) < 30:
        return None
    median = np.median(intervals)
    keep = (intervals >= 300) & (intervals <= 1500) & (np.abs(intervals - median) / median < .25)
    accepted = intervals[keep]
    differences = np.diff(intervals)[keep[1:] & keep[:-1]]
    if keep.mean() < .85 or len(differences) < 25:
        return None
    return dict(rmssd=float(np.sqrt(np.mean(differences ** 2))), sdnn=float(accepted.std()),
                intervals=accepted.tolist(), beatCount=len(peaks), rejected=int((~keep).sum()),
                valid=bool(abs(60000 / accepted.mean() - expected_bpm) < 12))
