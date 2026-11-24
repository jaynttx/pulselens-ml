import { mean, std, clamp, metrics, csv } from './signal.js';
import { plot } from './charts.js';

const $ = id => document.getElementById(id);
let method = 'pos', running = false, starting = false, stream = null, detector = null;
let analysisWorker = null, deepWorker = null, frameRequest = 0, runId = 0;
let rows = [], readings = [], pulseLog = [], observations = [];
let previousBox = null, lastVideoTime = -1, lastFrameAt = 0, lastFaceAt = 0;
let lastAnalysisAt = 0, startedAt = 0, deepBusy = false, analysisBusy = false;
let goodFrames = 0, lastResult = null, validStreak = 0, frameTimes = [], rawAccepted = [];
let lastDeepFrameAt = 0;
const crop = document.createElement('canvas'); crop.width = crop.height = 36;
const cropContext = crop.getContext('2d', { willReadFrequently: true });
const sample = document.createElement('canvas'); sample.width = sample.height = 64;
const sampleContext = sample.getContext('2d', { willReadFrequently: true });

function message(text, error = false) { $('notice').textContent = text; $('notice').classList.toggle('error', error); }
function displayInvalid(text) {
  $('bpm').textContent = '—'; $('reading-status').textContent = text; $('reading-status').classList.remove('good');
}
function chartReset() {
  plot($('wave')); plot($('spectrum')); plot($('trend'));
  $('quality').textContent = '—'; $('quality-fill').style.width = '0%'; $('peak-label').textContent = '— Hz';
}
function resetSignals() {
  rows = []; lastResult = null; goodFrames++; validStreak = 0; rawAccepted = [];
  lastAnalysisAt = 0; chartReset();
}
function resetSession() {
  resetSignals(); readings = []; pulseLog = []; previousBox = null; frameTimes = [];
  lastFrameAt = 0; lastFaceAt = 0; lastVideoTime = -1; lastDeepFrameAt = 0;
  ['min', 'average', 'max'].forEach(id => $(id).textContent = '—');
  $('timer').textContent = '00:00'; $('export').disabled = true;
}

document.querySelectorAll('[data-method]').forEach(button => button.addEventListener('click', () => {
  if (running || starting) return;
  method = button.dataset.method;
  document.querySelectorAll('[data-method]').forEach(b => { b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  $('method-description').textContent = method === 'pos'
    ? 'POS projects facial colour changes to reduce shared lighting noise. No pulse model is required.'
    : 'FacePhys is a pretrained state-space neural network. Its model assets load on first use; inference stays on your device.';
  message(method === 'deep' ? 'Deep learning selected. First start downloads the pretrained FacePhys model and runtime.' : 'Classical POS selected. Ready to start.');
}));

function updateView() {
  const route = ['monitor', 'benchmark', 'about'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'monitor';
  if (route !== 'monitor' && running) stop('Camera stopped while leaving the monitor. Export the session to keep it.');
  document.querySelectorAll('.view').forEach(view => view.hidden = view.id !== route);
  document.querySelectorAll('.nav-link').forEach(link => link.classList.toggle('active', link.hash === '#' + route));
  if (route === 'monitor') renderCharts();
}
window.addEventListener('hashchange', updateView);

async function createDetector() {
  const { FaceDetector, FilesetResolver } = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm');
  const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm');
  const modelAssetPath = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite';
  return FaceDetector.createFromOptions(vision, { baseOptions: { modelAssetPath, delegate: 'CPU' }, runningMode: 'VIDEO', minDetectionConfidence: 0.7 });
}

async function initializeDeep(token) {
  const worker = new Worker(new URL('./deep.worker.js', import.meta.url), { type: 'module' });
  deepWorker = worker;
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Model loading timed out. Check internet access or run npm run models.')), 90000);
    worker.onerror = () => { clearTimeout(timeout); reject(new Error('Deep learning runtime could not load. Check your connection.')); };
    worker.onmessage = ({ data }) => {
      if (data.type === 'ready') { clearTimeout(timeout); resolve(); }
      if (data.type === 'error') { clearTimeout(timeout); reject(new Error(data.message)); }
    };
    worker.postMessage({ type: 'init' });
  });
  if (token !== runId) { worker.terminate(); return; }
  worker.onmessage = ({ data }) => {
    if (token !== runId || !running) return;
    deepBusy = false;
    if (data.type === 'error') { stop(); message('Deep learning error: ' + data.message, true); return; }
    if (data.type === 'pulse') {
      if (performance.now() - lastFaceAt > 250) return;
      append({ t: data.t, v: [data.value] });
      $('latency').textContent = `${data.latency.toFixed(1)} ms`;
    }
  };
  worker.onerror = () => { if (token === runId) { stop(); message('Deep inference failed. Restart the camera or select Classical POS.', true); } };
}

async function start() {
  if (running || starting) return;
  if (!$('consent').checked) { message('Please confirm consent for local webcam estimation before starting.', true); return; }
  if (!navigator.mediaDevices?.getUserMedia) { message('Camera access requires localhost or HTTPS in a compatible browser.', true); return; }
  starting = true; const token = ++runId;
  $('start').disabled = true; $('start').textContent = 'Preparing…'; $('stop').disabled = false;
  document.querySelectorAll('[data-method]').forEach(b => b.disabled = true);
  resetSession();
  message(method === 'deep' ? 'Loading the face detector and pretrained model. First use may take a minute.' : 'Loading the face detector. Your camera will request permission next.');
  try {
    if (!detector) detector = await createDetector();
    if (token !== runId) return;
    if (method === 'deep') await initializeDeep(token);
    if (token !== runId) return;
    const acquired = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: 'user' } });
    if (token !== runId) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired; $('video').srcObject = stream; await $('video').play();
    if (token !== runId) { acquired.getTracks().forEach(t => t.stop()); return; }
    $('overlay').width = $('video').videoWidth; $('overlay').height = $('video').videoHeight;
    $('placeholder').hidden = true; document.querySelector('.camera-stage').classList.add('live');
    $('capture-status').textContent = 'Camera live'; $('capture-status').classList.add('good');
    analysisWorker = new Worker(new URL('./analysis.worker.js', import.meta.url), { type: 'module' });
    analysisWorker.onmessage = ({ data }) => {
      if (token !== runId || !running) return;
      analysisBusy = false;
      if (data.error) { stop(); message(data.error, true); return; }
      if (data.id !== goodFrames || performance.now() - lastFaceAt > 250) return;
      applyResult(data.result);
    };
    analysisWorker.onerror = () => { stop(); message('Signal analysis failed. Please restart the camera.', true); };
    running = true; starting = false; startedAt = performance.now();
    deepBusy = false; analysisBusy = false;
    $('start').textContent = 'Camera running'; displayInvalid('Finding your face');
    message('Keep your face still and evenly lit. Building a 20-second signal window.');
    frameRequest = requestAnimationFrame(tick);
  } catch (error) {
    if (token === runId) { stop(); message(error.name === 'NotAllowedError' ? 'Camera permission was denied. Allow access in browser settings, then try again.' : error.message, true); }
  } finally { if (token === runId && !running) restoreControls(); }
}
function restoreControls() {
  starting = false; $('start').disabled = false; $('start').textContent = 'Start new session'; $('stop').disabled = true;
  document.querySelectorAll('[data-method]').forEach(b => b.disabled = false);
}
function stop(text = 'Camera stopped. You can export your session or start a new one.') {
  ++runId; running = false; starting = false; cancelAnimationFrame(frameRequest);
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  $('video').srcObject = null; deepWorker?.terminate(); analysisWorker?.terminate();
  deepWorker = null; analysisWorker = null; deepBusy = false; analysisBusy = false;
  $('placeholder').hidden = false; document.querySelector('.camera-stage').classList.remove('live');
  $('overlay').getContext('2d').clearRect(0, 0, $('overlay').width, $('overlay').height);
  $('capture-status').textContent = 'Camera off'; $('capture-status').classList.remove('good');
  displayInvalid('Session stopped'); restoreControls(); message(text);
}

function append(row) {
  rows.push(row);
  while (rows.length > 0 && row.t - rows[0].t > 20000) rows.shift();
  // One entry per acquired sample; bounded to 30 minutes to avoid unlimited memory.
  if (pulseLog.length < 54000) pulseLog.push({ ...row, v: [...row.v] });
  $('export').disabled = pulseLog.length === 0;
}

function tick(now) {
  if (!running) return;
  frameRequest = requestAnimationFrame(tick);
  if ($('video').currentTime === lastVideoTime || now - lastFrameAt < 30 || $('video').readyState < 2) return;
  lastVideoTime = $('video').currentTime;
  const dt = lastFrameAt ? (now - lastFrameAt) / 1000 : 1 / 30;
  lastFrameAt = now; frameTimes.push(now); while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();
  $('fps').textContent = `${frameTimes.length} FPS`;
  const sec = Math.floor((now - startedAt) / 1000);
  $('timer').textContent = `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`;
  try {
    if (dt > 0.25) resetSignals();
    const detected = detector.detectForVideo($('video'), now).detections;
    const context = $('overlay').getContext('2d'); context.clearRect(0, 0, $('overlay').width, $('overlay').height);
    if (detected.length !== 1) {
      if (rows.length) resetSignals(); previousBox = null;
      displayInvalid(detected.length ? 'One person at a time' : 'Face not detected');
      $('guidance').textContent = detected.length ? 'Only one person should be visible in the camera.' : 'Bring your face into view and face the camera.'; return;
    }
    const b = detected[0].boundingBox;
    const box = { x: b.originX, y: b.originY - b.height * 0.15, w: b.width, h: b.height * 1.15 };
    const motion = previousBox ? Math.hypot(box.x - previousBox.x, box.y - previousBox.y) / box.w + Math.abs(box.w - previousBox.w) / box.w : 0;
    previousBox = box;
    context.strokeStyle = motion > 0.04 ? '#d6a585' : '#bca0dc'; context.lineWidth = 2;
    context.strokeRect(box.x, box.y, box.w, box.h);
    // Sample three fixed subregions within the detected face, avoiding eyes/mouth.
    sampleContext.drawImage($('video'), box.x, box.y, box.w, box.h, 0, 0, 64, 64);
    const patches = [[21, 6, 22, 9], [9, 32, 13, 13], [42, 32, 13, 13]];
    const rgb = [0, 0, 0]; let count = 0, clipped = 0;
    for (const [x, y, w, h] of patches) {
      const pixels = sampleContext.getImageData(x, y, w, h).data;
      for (let i = 0; i < pixels.length; i += 4) { for (let k = 0; k < 3; k++) rgb[k] += pixels[i + k]; if (Math.max(...pixels.slice(i, i + 3)) > 250) clipped++; count++; }
    }
    for (let k = 0; k < 3; k++) rgb[k] /= count;
    const brightness = mean(rgb);
    if (motion > 0.055 || brightness < 35 || brightness > 235 || clipped / count > 0.25 || box.w < 70) {
      if (rows.length) resetSignals(); lastFaceAt = 0;
      displayInvalid(motion > 0.055 ? 'Hold still to measure' : 'Adjust distance / lighting');
      $('guidance').textContent = motion > 0.055 ? 'Movement detected. The signal window has been restarted.' : 'Move closer and use even, comfortable lighting without glare.'; return;
    }
    lastFaceAt = now;
    if (method === 'pos') append({ t: now, v: rgb });
    else if (!deepBusy) {
      cropContext.drawImage($('video'), box.x, box.y, box.w, box.h, 0, 0, 36, 36);
      const rgba = cropContext.getImageData(0, 0, 36, 36).data, pixels = new Float32Array(36 * 36 * 3);
      for (let i = 0; i < 36 * 36; i++) for (let k = 0; k < 3; k++) pixels[i * 3 + k] = rgba[i * 4 + k] / 255;
      const neuralDt = lastDeepFrameAt ? (now - lastDeepFrameAt) / 1000 : 1 / 30;
      lastDeepFrameAt = now;
      deepBusy = true; deepWorker.postMessage({ type: 'frame', pixels, dt: neuralDt, t: now }, [pixels.buffer]);
    }
    const duration = rows.length ? (now - rows[0].t) / 1000 : 0;
    if (duration < 12) {
      displayInvalid(`Collecting signal · ${Math.min(100, Math.floor(duration / 12 * 100))}%`);
      $('guidance').textContent = 'Stay still while the signal builds. A longer observation gives better frequency resolution.';
    }
    if (duration >= 12 && !analysisBusy && now - lastAnalysisAt > 1000) {
      lastAnalysisAt = now; analysisBusy = true; goodFrames++;
      analysisWorker.postMessage({ method, rows, id: goodFrames });
    }
  } catch (error) { stop(); message('Capture error: ' + error.message, true); }
}

function applyResult(result) {
  lastResult = result;
  if (!result) { displayInvalid('No periodic signal yet'); validStreak = 0; return; }
  const q = Math.round(result.quality * 100);
  $('quality').textContent = `${q}%`; $('quality-fill').style.width = `${q}%`;
  $('peak-label').textContent = `${result.frequency.toFixed(2)} Hz`;
  $('wave-label').textContent = method === 'pos' ? 'POS · 20 s rolling window' : 'FacePhys · predicted pulse';
  validStreak = result.valid ? validStreak + 1 : 0;
  if (validStreak >= 3) {
    // Median of recent accepted values suppresses isolated fluctuations without
    // manufacturing readings during low-quality intervals.
    rawAccepted.push(result.bpm); if (rawAccepted.length > 3) rawAccepted.shift();
    const recent = [...rawAccepted].sort((a, b) => a - b);
    const bpm = recent[Math.floor(recent.length / 2)];
    readings.push({ t: performance.now(), bpm, quality: result.quality, method });
    if (readings.length > 1800) readings.shift();
    $('bpm').textContent = bpm.toFixed(0); $('reading-status').textContent = 'Stable experimental estimate'; $('reading-status').classList.add('good');
    const values = readings.map(r => r.bpm);
    $('min').textContent = Math.min(...values).toFixed(0); $('average').textContent = mean(values).toFixed(0); $('max').textContent = Math.max(...values).toFixed(0);
    $('guidance').textContent = 'Your signal is periodic and stable. Keep the same position for a longer observation.';
    $('app-bpm').value = bpm.toFixed(1); $('benchmark-method').value = method;
  } else { displayInvalid(result.valid ? 'Confirming signal…' : 'Signal needs improvement'); $('guidance').textContent = 'Keep still and improve lighting. Unreliable estimates are hidden.'; }
  renderCharts();
}
function renderCharts() {
  plot($('wave'), lastResult?.waveform?.slice(-300) || []);
  plot($('spectrum'), lastResult?.powers || [], { min: 0, max: 1, bars: true });
  plot($('trend'), readings.slice(-180).map(r => r.bpm), { min: 40, max: 200 });
}

function download(filename, content) {
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('export').addEventListener('click', () => {
  const output = [['kind', 'seconds', 'method', 'bpm', 'quality', 'red_or_bvp', 'green', 'blue']];
  for (const row of pulseLog) output.push(['sample', ((row.t - startedAt) / 1000).toFixed(3), method, '', '', ...row.v]);
  for (const row of readings) output.push(['reading', ((row.t - startedAt) / 1000).toFixed(3), row.method, row.bpm.toFixed(2), row.quality.toFixed(4), '', '', '']);
  download(`pulselens-${method}-${new Date().toISOString().replaceAll(':', '-')}.csv`, csv(output));
});

function renderBenchmark() {
  const body = $('observations'); body.replaceChildren();
  observations.forEach((row, index) => {
    const tr = document.createElement('tr');
    [row.condition, row.method === 'pos' ? 'Classical POS' : 'FacePhys SSM', row.app, row.reference, (row.app - row.reference).toFixed(1)].forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
    const td = document.createElement('td'), button = document.createElement('button'); button.className = 'delete-row'; button.textContent = 'Remove';
    button.setAttribute('aria-label', `Remove observation ${index + 1}`);
    button.onclick = () => { observations.splice(index, 1); renderBenchmark(); }; td.append(button); tr.append(td); body.append(tr);
  });
  const summary = ['pos', 'deep'].map(m => {
    const result = metrics(observations.filter(r => r.method === m));
    return result ? `${m === 'pos' ? 'POS' : 'SSM'} · n=${result.n} · MAE ${result.mae.toFixed(2)} BPM · RMSE ${result.rmse.toFixed(2)} BPM · Bias ${result.bias.toFixed(2)} BPM` : null;
  }).filter(Boolean);
  $('benchmark-metrics').textContent = summary.join(' | ') || 'No observations yet. Measured results will appear here.';
  $('export-benchmark').disabled = !observations.length;
}
$('benchmark-form').addEventListener('submit', event => {
  event.preventDefault();
  const app = Number($('app-bpm').value), reference = Number($('ref-bpm').value);
  if (![app, reference].every(v => Number.isFinite(v) && v >= 30 && v <= 240)) return;
  observations.push({ condition: $('condition').value, method: $('benchmark-method').value, app, reference }); renderBenchmark();
});
$('export-benchmark').addEventListener('click', () => download('pulselens-benchmark.csv', csv([['condition', 'method', 'app_bpm', 'reference_bpm'], ...observations.map(r => [r.condition, r.method, r.app, r.reference])])));
$('start').addEventListener('click', start); $('stop').addEventListener('click', () => stop());
$('consent').addEventListener('change', () => { if (!$('consent').checked && (running || starting)) stop('Consent withdrawn. Camera and inference stopped.'); });
document.addEventListener('visibilitychange', () => { if (document.hidden && (running || starting)) stop('Session stopped because this tab became hidden.'); });
window.addEventListener('pagehide', () => { stream?.getTracks().forEach(track => track.stop()); deepWorker?.terminate(); analysisWorker?.terminate(); });
window.addEventListener('resize', renderCharts);
updateView();
