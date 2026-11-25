"""POS baseline with the reference project's window/filter/fusion settings.

Written with NumPy rather than the reference's TypeScript routines. This is
an analytical baseline, not a trained model or a medical-quality guarantee.
"""
import numpy as np
from .signals import resample, pos, estimate


def estimate_live_rgb(rows):
    """POS on a time-based, uniformly sampled window for live webcam capture.

    Keep estimate_rgb below as the pinned reference profile for comparisons.
    A frame count is not a duration when HTTP/camera frames arrive unevenly.
    """
    if len(rows) < 2:
        return None
    recent = [row for row in rows if rows[-1][0] - row[0] <= 20000]
    elapsed = (recent[-1][0] - recent[0][0]) / 1000
    if elapsed < 10:
        return None
    intervals = np.diff([row[0] for row in recent]) / 1000
    if np.any(intervals <= 0) or (len(recent) - 1) / elapsed < 8 or intervals.max() > .5:
        return None
    # Overlap-normalized POS avoids amplifying the middle of a rolling buffer;
    # detrending and a tapered spectrum suppress window-edge / slow-light drift.
    result = estimate(pos(resample(recent)))
    if result:
        result['windowSeconds'] = elapsed
    return result


def extract_pos(rgb):
    rgb = np.asarray(rgb, dtype=float)
    output = np.zeros(len(rgb))
    for start in range(len(rgb) - 47):
        block = rgb[start:start + 48]
        average = block.mean(axis=0)
        if np.any(average < 1):
            continue
        r, g, b = (block / average).T
        first, second = g - b, g + b - 2 * r
        spread = first.std()
        ratio = spread / second.std() if second.std() > 1e-10 else 0
        projected = first + ratio * second
        output[start:start + 48] += (projected - first.mean()) / max(spread, 1e-10) / 48
    return output


def local_detrend(values, window=60):
    values = np.asarray(values, dtype=float)
    total = np.r_[0., np.cumsum(values)]
    indices = np.arange(len(values))
    left = np.maximum(0, indices - window // 2)
    right = np.minimum(len(values), indices + (window + 1) // 2)
    return values - (total[right] - total[left]) / (right - left)


def estimate_rgb(rows):
    rows = rows[-256:]
    if len(rows) < 150:
        return None
    elapsed = (rows[-1][0] - rows[0][0]) / 1000
    if elapsed <= 0:
        return None
    fs = (len(rows) - 1) / elapsed
    rgb = np.asarray([row[1] for row in rows], dtype=float)
    if not np.isfinite(rgb).all() or fs <= 2 * 200 / 60:
        return None
    pulse = local_detrend(extract_pos(rgb))
    nfft = 1 << (len(pulse) - 1).bit_length()
    frequency = np.fft.rfftfreq(nfft, 1 / fs)
    passband = (frequency >= 40 / 60) & (frequency <= 200 / 60)
    transformed = np.fft.rfft(pulse, nfft)
    transformed[~passband] = 0
    filtered = np.fft.irfft(transformed, nfft)[:len(pulse)]
    magnitude = np.abs(np.fft.rfft(filtered * np.hamming(len(filtered)), nfft))
    bins = np.flatnonzero(passband)
    if not len(bins) or filtered.std() < 1e-10:
        return None
    peak = bins[np.argmax(magnitude[bins])]
    spectral = None
    if magnitude[peak] >= 1e-10 and magnitude[peak] >= 2 * np.median(magnitude[bins]):
        refined = float(peak)
        if bins[0] < peak < bins[-1]:
            a, b, c = magnitude[peak - 1:peak + 2]
            if abs(a - 2 * b + c) > 1e-10:
                refined += .5 * (a - c) / (a - 2 * b + c)
        spectral = (refined * fs / nfft, float(magnitude[peak] ** 2 / max((magnitude[bins] ** 2).sum(), 1e-12)))
    energy = float(filtered @ filtered)
    low = max(1, int(fs / (200 / 60))); high = min(len(filtered) - 1, int(np.ceil(fs / (40 / 60))))
    temporal = None
    if high > low and energy > 1e-10:
        lags = np.arange(low, high + 1)
        correlations = np.array([filtered[:-lag] @ filtered[lag:] / energy for lag in lags])
        best = int(correlations.argmax()); lag = int(lags[best]); confidence = float(correlations[best])
        refined = float(lag)
        if low < lag < high:
            # Reference interpolation uses the same truncated overlap for neighbours.
            count = len(filtered) - lag - 1
            before = filtered[:count] @ filtered[lag - 1:lag - 1 + count] / energy
            after = filtered[:count] @ filtered[lag + 1:lag + 1 + count] / energy
            denominator = before - 2 * confidence + after
            if abs(denominator) > 1e-10:
                refined += .5 * (before - after) / denominator
        if confidence > 0:
            temporal = (fs / refined, confidence)
    candidates = [candidate for candidate in [spectral, temporal] if candidate]
    if not candidates:
        return None
    hz, confidence = max(candidates, key=lambda candidate: candidate[1])
    if spectral and temporal and abs(spectral[0] - temporal[0]) / max(spectral[0], temporal[0]) < .1:
        hz = (spectral[0] + temporal[0]) / 2
        confidence = max(spectral[1], temporal[1])
    bpm = float(hz * 60)
    quality = float(np.clip(confidence, 0, 1)); valid = confidence >= .08 and 40 <= bpm <= 200
    power = magnitude[bins] ** 2
    return dict(bpm=bpm, frequency=float(hz), quality=quality, valid=bool(valid),
                waveform=(filtered / max(filtered.std(), 1e-12)).tolist(),
                frequencies=frequency[bins].tolist(), powers=(power / max(power.max(), 1e-12)).tolist(),
                reason='Reference POS frequency estimates accepted.' if valid else 'No consistent pulse frequency yet.')
