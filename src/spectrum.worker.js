// Independent learned-quality / spectrum worker: does not block frame inference.
importScripts('./model-runtime.js');
let api, qualityModel, spectrumModel;
self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      const runtime = await self.PulseModels;
      api = runtime.api;
      [qualityModel, spectrumModel] = await Promise.all([runtime.compile('sqi_model.tflite'), runtime.compile('psd_model.tflite')]);
      self.postMessage({ type: 'ready' }); return;
    }
    if (data.type !== 'analyze' || !qualityModel) return;
    if (data.samples.length !== 450 || data.samples.some(v => !Number.isFinite(v))) throw new Error('Expected 450 finite raw neural samples.');
    if (!Number.isFinite(data.dt) || data.dt <= 0) throw new Error('Invalid neural sample timing.');
    const input = new api.Tensor(new Float32Array(data.samples), [1, 450]);
    let quality = [], spectrum = [];
    try {
      quality = qualityModel.run([input]); spectrum = spectrumModel.run([input]);
      const sqi = quality[0].toTypedArray()[0];
      // Training assumes 30 Hz; elapsed time restores the real camera frequency.
      const bpm = spectrum[0].toTypedArray()[0] / (30 * data.dt);
      if (!Number.isFinite(sqi) || !Number.isFinite(bpm)) throw new Error('Non-finite learned analysis output.');
      self.postMessage({ type: 'analysis', id: data.id, epoch: data.epoch, sqi, bpm,
        frequencies: Array.from(spectrum[1].toTypedArray(), v => v / (30 * data.dt)),
        powers: Array.from(spectrum[2].toTypedArray()) });
    } finally {
      input.delete(); quality.forEach(t => t.delete()); spectrum.forEach(t => t.delete());
    }
  } catch (error) { self.postMessage({ type: 'error', message: error.message }); }
};
