// FacePhys tensor contract inspected at the pinned PulseLens revision below.
// The pretrained model is third-party work, not trained by this project.
const REVISION = 'e7bdd94cbe686611eb37f817bbdf48b8d770b057';
const REMOTE = `https://raw.githubusercontent.com/xan-antx/pulseLens/${REVISION}/apps/pulselens-deep/models/`;
const RUNTIME = 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/';
// Input 0 = dt, input 1 = RGB crop; remaining inputs carry the hidden state.
const OUTPUT_FOR_INPUT = [null, null, 1, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 23,
  14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 34, 25, 26, 27, 28, 29, 30, 31, 32, 33,
  35, 42, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 2];
let library, model, tensors = [];

async function asset(name) {
  const local = await fetch(new URL(`../models/${name}`, import.meta.url));
  if (local.ok) return local;
  const remote = await fetch(REMOTE + name);
  if (!remote.ok) throw new Error(`Cannot load ${name}. Run npm run models with internet access.`);
  return remote;
}

async function initialize() {
  library = await import(RUNTIME + '+esm');
  // The runtime resolves WASM relative to its bundle in some browsers.
  const originalFetch = self.fetch;
  self.fetch = (input, init) => originalFetch(typeof input === 'string' && input.endsWith('.wasm') ? RUNTIME + 'wasm/' + input.split('/').pop() : input, init);
  try { await library.loadLiteRt(RUNTIME + 'wasm/'); }
  finally { self.fetch = originalFetch; }
  const [modelResponse, stateResponse] = await Promise.all([asset('model.tflite'), asset('state.gz')]);
  const buffer = await modelResponse.arrayBuffer();
  const state = await new Response(stateResponse.body.pipeThrough(new DecompressionStream('gzip'))).json();
  const url = URL.createObjectURL(new Blob([buffer]));
  try { model = await library.loadAndCompile(url, { accelerator: 'wasm' }); }
  finally { URL.revokeObjectURL(url); }
  const inputs = model.getInputDetails();
  if (inputs.length !== OUTPUT_FOR_INPUT.length) throw new Error('FacePhys input contract mismatch. Model assets must use the pinned revision.');
  tensors = inputs.map((meta, index) => {
    const size = meta.shape.reduce((a, b) => a * b, 1);
    const data = index === 0 ? [1 / 30] : index > 1 && state[meta.name] ? state[meta.name].flat(Infinity) : Array(size).fill(0);
    if (data.length !== size) throw new Error(`Invalid initial state: ${meta.name}`);
    return new library.Tensor(new Float32Array(data), meta.shape);
  });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'init') {
      await initialize(); self.postMessage({ type: 'ready' }); return;
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
    self.postMessage({ type: 'pulse', value, t: data.t, latency: performance.now() - started });
  } catch (error) { self.postMessage({ type: 'error', message: error.message }); }
};
