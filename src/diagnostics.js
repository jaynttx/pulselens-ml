// Exercises actual pretrained model inference without accessing a camera.
// Synthetic test input is used only for diagnostics, never as a heart reading.
export function checkNeuralRuntime(progress = () => {}, WorkerClass = Worker) {
  const worker = new WorkerClass(new URL('./deep.worker.js', import.meta.url));
  const spectrum = new WorkerClass(new URL('./spectrum.worker.js', import.meta.url));
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('Model check timed out. Check asset access and browser support.')), 90000);
    const pulse = [];
    let frames = 0, projected = false, ready = 0, settled = false;
    function finish(error) { if (settled) return; settled = true; clearTimeout(timeout); worker.terminate(); spectrum.terminate(); error ? reject(error) : resolve({ inference: true, projection: projected, sqi: true, psd: true, frames }); }
    function sendFrame() {
      worker.postMessage({ type: 'frame', pixels: new Float32Array(36 * 36 * 3).fill(0.5), dt: 1 / 30, t: frames * 1000 / 30, epoch: 0 });
    }
    worker.onerror = spectrum.onerror = event => finish(new Error(event.message || 'Neural worker could not start.'));
    worker.onmessage = spectrum.onmessage = ({ data }) => {
      if (data.type === 'error') return finish(new Error(data.message));
      if (data.type === 'progress') return progress(data.message);
      if (data.type === 'ready' && ++ready === 2) { progress('Models loaded. Testing neural inference and feature projection…'); sendFrame(); }
      if (data.type === 'pulse') {
        if (!Number.isFinite(data.value)) return finish(new Error('Non-finite neural pulse output.'));
        pulse.push(data.value);
        projected ||= !!data.visualization;
        if (++frames < 480) sendFrame();
        else {
          if (!projected) return finish(new Error('Model projection produced no features.'));
          progress('480 recurrent frames passed. Testing learned SQI and PSD on those model outputs…');
          spectrum.postMessage({ type: 'analyze', id: 1, dt: 1 / 30, samples: pulse.slice(-450) });
        }
      }
      if (data.type === 'analysis') {
        if (!Number.isFinite(data.sqi) || !Number.isFinite(data.bpm)) return finish(new Error('Invalid SQI/PSD output.'));
        finish();
      }
    };
    worker.postMessage({ type: 'init' });
    spectrum.postMessage({ type: 'init' });
  });
}
