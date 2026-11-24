// FacePhys tensor contract inspected at the pinned PulseLens revision below.
// The pretrained model is third-party work, not trained by this project.
const REVISION = 'e7bdd94cbe686611eb37f817bbdf48b8d770b057';
const REMOTE = `https://raw.githubusercontent.com/xan-antx/pulseLens/${REVISION}/apps/pulselens-deep/models/`;
const RUNTIME = 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/';
// Input 0 = dt, input 1 = RGB crop; remaining inputs carry the hidden state.
const OUTPUT_FOR_INPUT = [null, null, 1, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 23,
  14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 34, 25, 26, 27, 28, 29, 30, 31, 32, 33,
  35, 42, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 2];
const PROJECTION_INPUTS = [4, 5, 12, 15, 16, 23, 26, 27, 34, 37, 38, 40, 41, 46];
const PROJECTION_KEYS = ['fm1', 'fm2', 'fm3', 'fm4', 'proj', 'ssm1', 'ssm2', 'ssm3', 'ssm4'];
let library, model, projection, sqiModel, psdModel, initialState, metadata, tensors = [], frame = 0;

async function asset(name) {
  try {
    const local = await fetch(new URL(`../models/${name}`, self.location.href));
    if (local.ok) return local;
  } catch { /* Cached remote assets can still work without local weights. */ }
  const remote = await fetch(REMOTE + name);
  if (!remote.ok) throw new Error(`Cannot load ${name}. Run npm run models with internet access.`);
  return remote;
}

async function initialize() {
  self.postMessage({ type: 'progress', message: 'Loading the neural WASM runtime…' });
  library = await import(RUNTIME + '+esm');
  // The runtime resolves WASM relative to its bundle in some browsers.
  const originalFetch = self.fetch;
  self.fetch = (input, init) => originalFetch(typeof input === 'string' && input.endsWith('.wasm') ? RUNTIME + 'wasm/' + input.split('/').pop() : input, init);
  try { await library.loadLiteRt(RUNTIME + 'wasm/'); }
  finally { self.fetch = originalFetch; }
  async function compile(name) {
    const buffer = await (await asset(name)).arrayBuffer();
    const url = URL.createObjectURL(new Blob([buffer]));
    try { return await library.loadAndCompile(url, { accelerator: 'wasm' }); }
    finally { URL.revokeObjectURL(url); }
  }
  self.postMessage({ type: 'progress', message: 'Loading pretrained FacePhys, projection, quality and spectrum models…' });
  [model, projection, sqiModel, psdModel] = await Promise.all([
    compile('model.tflite'), compile('proj.tflite'), compile('sqi_model.tflite'), compile('psd_model.tflite')
  ]);
  const stateResponse = await asset('state.gz');
  self.postMessage({ type: 'progress', message: 'Initializing recurrent neural state…' });
  initialState = await new Response(stateResponse.body.pipeThrough(new DecompressionStream('gzip'))).json();
  const inputs = model.getInputDetails();
  if (inputs.length !== OUTPUT_FOR_INPUT.length) throw new Error('FacePhys input contract mismatch. Model assets must use the pinned revision.');
  metadata = inputs; resetState();
}

function resetState() {
  tensors.forEach(t => t.delete()); frame = 0;
  tensors = metadata.map((meta, index) => {
    const size = meta.shape.reduce((a, b) => a * b, 1);
    const data = index === 0 ? [1 / 30] : index > 1 && initialState[meta.name] ? initialState[meta.name].flat(Infinity) : Array(size).fill(0);
    if (data.length !== size) throw new Error(`Invalid initial state: ${meta.name}`);
    return new library.Tensor(new Float32Array(data), meta.shape);
  });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      await initialize(); self.postMessage({ type: 'ready' }); return;
    }
    if (data.type === 'reset' && model) { resetState(); return; }
    if (data.type === 'analyze' && sqiModel) {
      if (data.samples.length !== 450 || data.samples.some(v => !Number.isFinite(v))) throw new Error('SQI requires 450 finite pulse samples.');
      const input = new library.Tensor(new Float32Array(data.samples), [1, 450]);
      let quality = [], spectrum = [];
      try {
        quality = sqiModel.run([input]); spectrum = psdModel.run([input]);
        const sqi = quality[0].toTypedArray()[0], bpm = spectrum[0].toTypedArray()[0];
        if (!Number.isFinite(sqi) || !Number.isFinite(bpm)) throw new Error('Invalid learned quality/PSD output.');
        self.postMessage({ type: 'analysis', id: data.id, sqi, bpm });
      } finally { input.delete(); quality.forEach(t => t.delete()); spectrum.forEach(t => t.delete()); }
      return;
    }
    if (data.type !== 'frame' || !model) return;
    const started = performance.now();
    tensors[0].delete(); tensors[1].delete();
    tensors[0] = new library.Tensor(new Float32Array([data.dt]), [1]);
    tensors[1] = new library.Tensor(data.pixels, [1, 1, 36, 36, 3]);
    const outputs = model.run(tensors);
    const value = outputs[0].toTypedArray()[0];
    const retained = new Set();
    for (let i = 2; i < tensors.length; i++) {
      const output = OUTPUT_FOR_INPUT[i];
      if (!outputs[output]) throw new Error(`Missing model output ${output}`);
      tensors[i].delete(); tensors[i] = outputs[output]; retained.add(output);
    }
    outputs.forEach((tensor, i) => { if (!retained.has(i)) tensor.delete(); });
    if (!Number.isFinite(value)) throw new Error('Model produced a non-finite pulse value. Restart the session.');
    let visualization = null;
    if (++frame % 5 === 0) {
      const projected = projection.run(PROJECTION_INPUTS.map(i => tensors[i]));
      try { visualization = Object.fromEntries(PROJECTION_KEYS.map((key, i) => [key, Array.from(projected[i].toTypedArray())])); }
      finally { projected.forEach(t => t.delete()); }
    }
    self.postMessage({ type: 'pulse', value, t: data.t, epoch: data.epoch, visualization, latency: performance.now() - started });
  } catch (error) { self.postMessage({ type: 'error', message: error.message }); }
};
