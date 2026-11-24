import { mean, std, detrend, resample } from './signal.js';

export function bandpass(values, fs = 30, low = 0.67, high = 3.33) {
  if (!values.length) return [];
  const a = Math.exp(-2 * Math.PI * low / fs), b = 1 - Math.exp(-2 * Math.PI * high / fs);
  let previous = values[0], hp = 0, lp = 0;
  return values.map(value => { hp = a * (hp + value - previous); previous = value; lp += b * (hp - lp); return lp; });
}

// Respiratory proxy: low-frequency variation in facial green intensity. This
// is exploratory and cannot distinguish breathing from periodic lighting/motion.
export function breathing(rows, fs = 30) {
  if (rows.length < 2 || rows.at(-1).t - rows[0].t < 30000) return null;
  const rgb = resample(rows, fs), green = rgb.map(v => v[1]);
  if (std(green) < 1e-8) return null;
  const signal = bandpass(detrend(green), fs, 0.08, 0.6).slice(fs * 3);
  const n = signal.length, powers = [], rates = [];
  for (let rate = 6; rate <= 30; rate += 0.25) {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) {
      const value = signal[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
      const phase = 2 * Math.PI * rate / 60 * i / fs;
      re += value * Math.cos(phase); im += value * Math.sin(phase);
    }
    powers.push(re * re + im * im); rates.push(rate);
  }
  const peak = powers.indexOf(Math.max(...powers)), rate = rates[peak], total = powers.reduce((a, b) => a + b, 0);
  const concentration = powers.reduce((sum, p, i) => sum + (Math.abs(rates[i] - rate) <= 2 ? p : 0), 0) / Math.max(total, 1e-12);
  return { rate, quality: concentration, valid: concentration > 0.55 && peak > 0 && peak < rates.length - 1 };
}

export function variability(pulse, fs = 30, expectedBpm) {
  if (pulse.length / fs < 55 || !Number.isFinite(expectedBpm)) return null;
  const filtered = bandpass(detrend(pulse), fs), skip = fs * 3;
  const spread = std(filtered.slice(skip)); if (spread < 1e-10) return null;
  const refractory = Math.max(0.3, 60 / expectedBpm * 0.65) * fs;
  const peaks = [];
  for (let i = skip + 1; i < filtered.length - 1; i++) {
    if (filtered[i] < spread * 0.15 || filtered[i] <= filtered[i - 1] || filtered[i] < filtered[i + 1]) continue;
    const curvature = filtered[i - 1] - 2 * filtered[i] + filtered[i + 1];
    const offset = curvature ? Math.max(-0.5, Math.min(0.5, 0.5 * (filtered[i - 1] - filtered[i + 1]) / curvature)) : 0;
    const point = { index: i + offset, value: filtered[i] };
    if (peaks.length && point.index - peaks.at(-1).index < refractory) {
      if (point.value > peaks.at(-1).value) peaks[peaks.length - 1] = point;
    } else peaks.push(point);
  }
  const intervals = peaks.slice(1).map((p, i) => (p.index - peaks[i].index) * 1000 / fs);
  if (intervals.length < 30) return null;
  const sorted = [...intervals].sort((a, b) => a - b), median = sorted[Math.floor(sorted.length / 2)];
  const valid = intervals.map(v => v >= 300 && v <= 1500 && Math.abs(v - median) / median < 0.25);
  const accepted = intervals.filter((_, i) => valid[i]), differences = [];
  // Do not bridge rejected intervals: only originally adjacent accepted pairs.
  for (let i = 1; i < intervals.length; i++) if (valid[i] && valid[i - 1]) differences.push(intervals[i] - intervals[i - 1]);
  if (accepted.length / intervals.length < 0.85 || differences.length < 25) return null;
  return { rmssd: Math.sqrt(mean(differences.map(v => v * v))), sdnn: std(accepted),
    intervals: accepted, beatCount: peaks.length, rejected: intervals.length - accepted.length,
    valid: Math.abs(60000 / mean(accepted) - expectedBpm) < 12 };
}
