// All routines are independent of browser APIs and can be tested in Node.
export const mean = a => a.reduce((s, v) => s + v, 0) / (a.length || 1);
export const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Zero-phase, second-order Butterworth high/low passes. Remove out-of-band
// illumination drift before *both* spectral and time-domain quality checks.
export function bandpass(values, fs = 30, low = 0.67, high = 3.33) {
  if (values.length < 4) return [...values];
  function pass(input, cutoff, highpass) {
    const w = 2 * Math.PI * cutoff / fs, c = Math.cos(w), alpha = Math.sin(w) / Math.SQRT2;
    const a0 = 1 + alpha, a1 = -2 * c / a0, a2 = (1 - alpha) / a0;
    const b0 = (highpass ? (1 + c) : (1 - c)) / (2 * a0);
    const b1 = (highpass ? -(1 + c) : (1 - c)) / a0, b2 = b0;
    let z1 = 0, z2 = 0;
    return input.map(x => { const y = b0 * x + z1; z1 = b1 * x - a1 * y + z2; z2 = b2 * x - a2 * y; return y; });
  }
  const pad = Math.min(values.length - 1, Math.ceil(fs * 3));
  const extended = [...values.slice(1, pad + 1).reverse().map(v => 2 * values[0] - v), ...values,
    ...values.slice(-pad - 1, -1).reverse().map(v => 2 * values.at(-1) - v)];
  const forward = pass(pass(extended, low, true), high, false);
  const backward = pass(pass(forward.reverse(), low, true), high, false).reverse();
  return backward.slice(pad, pad + values.length);
}

export function detrend(a) {
  const n = a.length, center = (n - 1) / 2, m = mean(a);
  let numerator = 0, denominator = 0;
  for (let i = 0; i < n; i++) { numerator += (i - center) * (a[i] - m); denominator += (i - center) ** 2; }
  const slope = denominator ? numerator / denominator : 0;
  return a.map((v, i) => v - m - slope * (i - center));
}

// Uniform sampling avoids interpreting variable webcam timestamps as fixed FPS.
export function resample(rows, fs = 30) {
  if (rows.length < 2) return [];
  const end = rows.at(-1).t, start = rows[0].t, step = 1000 / fs;
  const output = []; let j = 1;
  for (let t = start; t <= end; t += step) {
    while (j < rows.length - 1 && rows[j].t < t) j++;
    const left = rows[j - 1], right = rows[j];
    const alpha = clamp((t - left.t) / Math.max(right.t - left.t, 1e-8), 0, 1);
    output.push(left.v.map((x, k) => x + (right.v[k] - x) * alpha));
  }
  return output;
}

export function pos(rgb, fs = 30) {
  const n = rgb.length, size = Math.round(1.6 * fs), output = Array(n).fill(0), counts = Array(n).fill(0);
  if (n < size) return output;
  for (let start = 0; start <= n - size; start++) {
    const window = rgb.slice(start, start + size);
    const average = [0, 1, 2].map(k => mean(window.map(v => v[k])));
    if (average.some(v => v < 1e-7)) continue;
    const normalized = window.map(v => v.map((x, k) => x / average[k]));
    const s1 = normalized.map(v => v[1] - v[2]);
    const s2 = normalized.map(v => v[1] + v[2] - 2 * v[0]);
    const alpha = std(s1) / Math.max(std(s2), 1e-9);
    const h = s1.map((v, i) => v + alpha * s2[i]), m = mean(h);
    for (let k = 0; k < size; k++) { output[start + k] += h[k] - m; counts[start + k]++; }
  }
  return detrend(output.map((v, i) => v / Math.max(counts[i], 1)));
}

// Iterative radix-2 FFT; padding permits peak interpolation without claiming
// more physical frequency resolution than the observation window provides.
export function powerSpectrum(values, fs = 30) {
  let n = 1;
  while (n < values.length * 4) n *= 2;
  const real = new Float64Array(n), imaginary = new Float64Array(n);
  real.set(values);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) [real[i], real[j]] = [real[j], real[i]];
  }
  for (let span = 2; span <= n; span *= 2) {
    for (let start = 0; start < n; start += span) {
      for (let k = 0; k < span / 2; k++) {
        const angle = -2 * Math.PI * k / span, left = start + k, right = left + span / 2;
        const re = real[right] * Math.cos(angle) - imaginary[right] * Math.sin(angle);
        const im = real[right] * Math.sin(angle) + imaginary[right] * Math.cos(angle);
        real[right] = real[left] - re; imaginary[right] = imaginary[left] - im;
        real[left] += re; imaginary[left] += im;
      }
    }
  }
  const frequencies = [], powers = [];
  for (let k = Math.ceil(40 / 60 * n / fs); k <= Math.floor(200 / 60 * n / fs); k++) {
    frequencies.push(k * fs / n); powers.push(real[k] ** 2 + imaginary[k] ** 2);
  }
  return { frequencies, powers, step: fs / n };
}

export function estimate(pulse, fs = 30) {
  if (pulse.length < fs * 5 || pulse.some(v => !Number.isFinite(v))) return null;
  const clean = bandpass(detrend(pulse), fs), spread = std(clean);
  if (spread < 1e-10) return null;
  const x = clean.map(v => v / spread);
  const taper = x.map((v, i) => v * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (x.length - 1))));
  const { frequencies, powers, step } = powerSpectrum(taper, fs);
  let peak = powers.indexOf(Math.max(...powers));
  let frequency = frequencies[peak];
  if (peak > 0 && peak < powers.length - 1) {
    const [a, b, c] = powers.slice(peak - 1, peak + 2), d = a - 2 * b + c;
    if (Math.abs(d) > 1e-10) frequency += clamp(0.5 * (a - c) / d, -1, 1) * step;
  }
  const total = powers.reduce((a, b) => a + b, 0);
  const signal = powers.reduce((s, v, i) => s + (Math.abs(frequencies[i] - frequency) < 0.12 ? v : 0), 0);
  const concentration = total > 0 ? signal / total : 0;
  const maxLag = Math.min(x.length - 1, Math.ceil(fs * 60 / 40));
  let bestLag = 0, bestCorrelation = -1;
  for (let lag = Math.floor(fs * 60 / 200); lag <= maxLag; lag++) {
    let dot = 0, a = 0, b = 0;
    for (let i = lag; i < x.length; i++) { dot += x[i] * x[i - lag]; a += x[i] ** 2; b += x[i - lag] ** 2; }
    const correlation = dot / Math.sqrt(Math.max(a * b, 1e-12));
    // Limit time-domain confirmation to the spectral peak neighborhood to avoid
    // accepting a multiple-period harmonic as a separate independent estimate.
    if (Math.abs(fs / lag - frequency) < 0.18 && correlation > bestCorrelation) { bestLag = lag; bestCorrelation = correlation; }
  }
  const agreement = bestLag && bestCorrelation > 0.3;
  const quality = clamp(concentration * Math.max(0, bestCorrelation), 0, 1);
  const valid = concentration > 0.45 && !!agreement && quality > 0.38;
  const reason = concentration <= 0.45 ? 'Pulse peak is not distinct from noise.' : !agreement ? 'Pulse timing is not periodic yet.' : quality <= 0.38 ? 'Pulse is present but too noisy for a reliable estimate.' : 'Periodic pulse detected.';
  return { bpm: frequency * 60, frequency, quality, valid, reason, concentration,
    frequencies, powers: powers.map(v => v / Math.max(...powers, 1e-12)), waveform: x, correlation: bestCorrelation };
}

export function metrics(rows) {
  if (!rows.length) return null;
  const e = rows.map(r => r.app - r.reference);
  return { n: e.length, mae: mean(e.map(Math.abs)), rmse: Math.sqrt(mean(e.map(v => v * v))), bias: mean(e) };
}

export function csv(rows) {
  // Spreadsheet formula escaping applies only to strings, preserving numeric negatives.
  return rows.map(row => row.map(value => {
    let s = String(value ?? '');
    if (typeof value === 'string' && /^[=+\-@\t\r]/.test(s)) s = "'" + s;
    return '"' + s.replaceAll('"', '""') + '"';
  }).join(',')).join('\r\n');
}
