import { resample, pos, estimate } from './signal.js';
import { breathing, variability } from './vitals.js';
self.onmessage = ({ data }) => {
  try {
    const end = data.rows.at(-1)?.t || 0;
    const samples = resample(data.rows.filter(row => end - row.t <= 20000), 30);
    const pulse = data.method === 'pos' ? pos(samples) : samples.map(v => v[0]);
    const result = estimate(pulse);
    if (result) {
      const whole = resample(data.rows, 30);
      const longPulse = data.method === 'pos' ? pos(whole) : whole.map(v => v[0]);
      result.variability = result.valid ? variability(longPulse, 30, result.bpm) : null;
      result.breathing = breathing(data.rgbRows || [], 30);
      if (data.method === 'deep') result.modelInput = samples.slice(-450).map(v => v[0]);
    }
    self.postMessage({ id: data.id, result });
  } catch (e) { self.postMessage({ id: data.id, error: e.message }); }
};
