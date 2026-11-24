// FacePhys tensor contract inspected at the pinned PulseLens revision below.
// The pretrained model is third-party work, not trained by this project.
importScripts('./model-runtime.js');
// Input 0 = dt, input 1 = RGB crop; remaining inputs carry the hidden state.
const OUTPUT_FOR_INPUT = [null, null, 1, 12, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 23,
  14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 34, 25, 26, 27, 28, 29, 30, 31, 32, 33,
  35, 42, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 2];
const PROJECTION_INPUTS = [4, 5, 12, 15, 16, 23, 26, 27, 34, 37, 38, 40, 41, 46];
const PROJECTION_KEYS = ['fm1', 'fm2', 'fm3', 'fm4', 'proj', 'ssm1', 'ssm2', 'ssm3', 'ssm4'];
let library, model, projection, initialState, metadata, tensors = [], frame = 0;

async function initialize() {
  self.postMessage({ type: 'progress', message: 'Loading the pretrained recurrent pulse model…' });
  const runtime = await self.PulseModels;
  library = runtime.api;
  [model, projection] = await Promise.all([runtime.compile('model.tflite'), runtime.compile('proj.tflite')]);
  const response = await runtime.asset('state.gz');
  initialState = await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();
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
