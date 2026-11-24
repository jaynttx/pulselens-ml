import { resample, pos, estimate } from './signal.js';
self.onmessage = ({ data }) => {
  try {
    const samples = resample(data.rows, 30);
    const pulse = data.method === 'pos' ? pos(samples) : samples.map(v => v[0]);
    self.postMessage({ id: data.id, result: estimate(pulse) });
  } catch (e) { self.postMessage({ id: data.id, error: e.message }); }
};
