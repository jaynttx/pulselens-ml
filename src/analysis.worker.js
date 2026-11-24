import { resample, pos, estimate, bandpass, std } from './signal.js';
import { breathing, variability } from './vitals.js';
self.onmessage = ({ data }) => {
  try {
    const end = data.rows.at(-1)?.t || 0;
    const samples = resample(data.rows.filter(row => end - row.t <= 20000), 30);
    const pulse = data.method === 'pos' ? pos(samples) : samples.map(v => v[0]);
    let result = estimate(pulse);
    if (data.method === 'deep' && data.rows.length >= 450) {
      // Learned postprocessing requires the last 450 *actual* model outputs.
      // Resampling them changes the distribution supplied to pretrained SQI.
      const raw = data.rows.slice(-450);
      const wave = bandpass(raw.map(row => row.v[0]));
      const scale = std(wave) || 1;
      result ||= { bpm: 0, frequency: 0, quality: 0, valid: false, powers: [], waveform: wave.map(v => v / scale) };
      result.modelInput = raw.map(row => row.v[0]);
      result.sampleDt = (raw.at(-1).t - raw[0].t) / (449 * 1000);
    }
    if (result) {
      const whole = resample(data.rows, 30);
      const longPulse = data.method === 'pos' ? pos(whole) : whole.map(v => v[0]);
      const beatRate = data.method === 'deep' ? data.referenceBpm : result.valid ? result.bpm : null;
      result.variability = beatRate ? variability(longPulse, 30, beatRate) : null;
      result.breathing = breathing(data.rgbRows || [], 30);
    }
    self.postMessage({ id: data.id, result });
  } catch (e) { self.postMessage({ id: data.id, error: e.message }); }
};
