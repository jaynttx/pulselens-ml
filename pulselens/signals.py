"""Timestamp resampling, POS, Butterworth filtering and FFT pulse estimation."""
import numpy as np


def resample(rows, fs=30):
    if len(rows) < 2:
        return np.empty((0, 1))
    times = np.asarray([r[0] for r in rows], dtype=float)
    values = np.asarray([r[1] for r in rows], dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Samples must be finite and timestamps strictly increasing.')
    grid = np.arange(times[0], times[-1] + 1e-7, 1000 / fs)
    return np.column_stack([np.interp(grid, times, values[:, k]) for k in range(values.shape[1])])


def detrend(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return values.copy()
    x = np.arange(len(values)) - (len(values) - 1) / 2
    centered = values - values.mean()
    return centered - x * (np.dot(x, centered) / max(np.dot(x, x), 1e-12))


def bandpass(values, fs=30, low=0.67, high=3.33):
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return values.copy()
    def apply(data, cutoff, highpass):
        w = 2 * np.pi * cutoff / fs
        c, alpha = np.cos(w), np.sin(w) / np.sqrt(2)
        a0 = 1 + alpha
        a1, a2 = -2 * c / a0, (1 - alpha) / a0
        b0 = (1 + c if highpass else 1 - c) / (2 * a0)
        b1 = (-(1 + c) if highpass else 1 - c) / a0
        out = np.empty(len(data)); z1 = z2 = 0.
        for i, value in enumerate(data):
            y = b0 * value + z1
            z1, z2 = b1 * value - a1 * y + z2, b0 * value - a2 * y
            out[i] = y
        return out
    pad = min(len(values) - 1, int(np.ceil(fs * 3)))
    extended = np.r_[2 * values[0] - values[1:pad + 1][::-1], values,
                     2 * values[-1] - values[-pad - 1:-1][::-1]]
    forward = apply(apply(extended, low, True), high, False)
    backward = apply(apply(forward[::-1], low, True), high, False)[::-1]
    return backward[pad:pad + len(values)]


def pos(rgb, fs=30):
    rgb = np.asarray(rgb, dtype=float)
    size = round(1.6 * fs)
    output = np.zeros(len(rgb)); counts = np.zeros(len(rgb))
    for start in range(len(rgb) - size + 1):
        window = rgb[start:start + size]
        average = window.mean(axis=0)
        if np.any(average < 1e-7):
            continue
        r, g, b = (window / average).T
        s1, s2 = g - b, g + b - 2 * r
        pulse = s1 + s1.std() / max(s2.std(), 1e-9) * s2
        output[start:start + size] += pulse - pulse.mean()
        counts[start:start + size] += 1
    return detrend(output / np.maximum(counts, 1))


def estimate(pulse, fs=30):
    pulse = np.asarray(pulse, dtype=float)
    if len(pulse) < fs * 5 or not np.isfinite(pulse).all():
        return None
    clean = bandpass(detrend(pulse), fs)
    spread = clean.std()
    if spread < 1e-10:
        return None
    wave = clean / spread
    nfft = 1 << (len(wave) * 4 - 1).bit_length()
    spectrum = np.abs(np.fft.rfft(wave * np.hanning(len(wave)), nfft)) ** 2
    frequency = np.fft.rfftfreq(nfft, 1 / fs)
    mask = (frequency >= 40 / 60) & (frequency <= 200 / 60)
    frequencies, powers = frequency[mask], spectrum[mask]
    peak = int(powers.argmax()); hz = frequencies[peak]
    if 0 < peak < len(powers) - 1:
        a, b, c = powers[peak - 1:peak + 2]
        if abs(a - 2 * b + c) > 1e-10:
            hz += np.clip(.5 * (a - c) / (a - 2 * b + c), -1, 1) * fs / nfft
    concentration = float(powers[np.abs(frequencies - hz) < .12].sum() / max(powers.sum(), 1e-12))
    correlation = -1.
    for lag in range(int(fs * 60 / 200), min(len(wave) - 1, int(np.ceil(fs * 60 / 40))) + 1):
        if abs(fs / lag - hz) >= .18:
            continue
        left, right = wave[lag:], wave[:-lag]
        correlation = max(correlation, float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12)))
    quality = float(np.clip(concentration * max(0, correlation), 0, 1))
    valid = concentration > .45 and correlation > .3 and quality > .38
    return dict(bpm=float(hz * 60), frequency=float(hz), quality=quality, valid=bool(valid),
                reason='Periodic pulse detected.' if valid else 'Use even lighting and keep your face in view.',
                frequencies=frequencies.tolist(), powers=(powers / max(powers.max(), 1e-12)).tolist(),
                waveform=wave.tolist(), concentration=concentration, correlation=correlation)
