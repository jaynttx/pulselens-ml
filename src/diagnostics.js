// Exercises actual pretrained model inference without accessing a camera.
// Synthetic test input is used only for diagnostics, never as a heart reading.
export function checkNeuralRuntime(progress = () => {}, WorkerClass = Worker) {
  const worker = new WorkerClass(new URL('./deep.worker.js', import.meta.url));
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => finish(new Error('Model check timed out. Check asset access and browser support.')), 90000);
    let frames = 0, projected = false;
    function finish(error) { clearTimeout(timeout); worker.terminate(); error ? reject(error) : resolve({ inference: true, projection: projected, sqi: true, psd: true }); }
    function sendFrame() {
      worker.postMessage({ type: 'frame', pixels: new Float32Array(36 * 36 * 3).fill(0.5), dt: 1 / 30, t: frames * 1000 / 30, epoch: 0 });
    }
    worker.onerror = event => finish(new Error(event.message || 'Neural worker could not start.'));
    worker.onmessage = ({ data }) => {
      if (data.type === 'error') return finish(new Error(data.message));
      if (data.type === 'progress') return progress(data.message);
      if (data.type === 'ready') { progress('Models loaded. Testing neural inference and feature projection…'); sendFrame(); }
      if (data.type === 'pulse') {
        if (!Number.isFinite(data.value)) return finish(new Error('Non-finite neural pulse output.'));
        projected ||= !!data.visualization;
        if (++frames < 6) sendFrame();
        else {
          if (!projected) return finish(new Error('Model projection produced no features.'));
          progress('Inference passed. Testing learned SQI and PSD…');
          worker.postMessage({ type: 'analyze', id: 1, samples: Array.from({ length: 450 }, (_, i) => Math.sin(2 * Math.PI * 1.2 * i / 30)) });
        }
      }
      if (data.type === 'analysis') {
        if (!Number.isFinite(data.sqi) || !Number.isFinite(data.bpm)) return finish(new Error('Invalid SQI/PSD output.'));
        finish();
      }
    };
    worker.postMessage({ type: 'init' });
  });
}
