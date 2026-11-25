import { mean, std, clamp, metrics, csv } from './ui-utils.js';
import { plot } from './charts.js';
import { clearPanel, heatmap, trajectory } from './visuals.js';
import { acquireCamera } from './camera.js';
import { ResearchPanels } from './insights.js';

const $ = id => document.getElementById(id);
const researchPanels = new ResearchPanels($);
let method = 'pos', running = false, starting = false, stream = null;
let frameRequest = 0, runId = 0, pythonSession = null, pythonBusy = false;
let rows = [], readings = [], pulseLog = [], observations = [];
let previousBox = null, lastVideoTime = -1, lastFrameAt = 0, lastFaceAt = 0;
let lastAnalysisAt = 0, startedAt = 0;
let goodFrames = 0, lastResult = null, validStreak = 0, frameTimes = [], rawAccepted = [];
let lastDeepFrameAt = 0;
let captureEpoch = 0, pendingAnalysis = null, learnedSqi = null, lastVisualization = null;
let offlinePreparing = false;
let stateHistory = [[], [], [], [], []], sessionMethod = method;
let missingSince = 0;
const crop = document.createElement('canvas'); crop.width = crop.height = 36;
const cropContext = crop.getContext('2d', { willReadFrequently: true });
const sample = document.createElement('canvas'); sample.width = sample.height = 64;
const sampleContext = sample.getContext('2d', { willReadFrequently: true });

function message(text, error = false) { $('notice').textContent = text; $('notice').classList.toggle('error', error); }
function displayInvalid(text) {
  $('bpm').textContent = '—'; $('reading-status').textContent = text; $('reading-status').classList.remove('good');
}
function chartReset() {
  plot($('wave')); plot($('spectrum')); plot($('trend'), readings.slice(-180).map(r => r.bpm), { min: 40, max: 200 });
  $('quality').textContent = '—'; $('quality-fill').style.width = '0%'; $('peak-label').textContent = '— Hz';
}
function resetSignals() {
  captureEpoch++; pendingAnalysis = null; learnedSqi = null;
  rows = []; lastResult = null; goodFrames++; validStreak = 0; rawAccepted = [];
  rgbRows = []; lastDeepFrameAt = 0; researchPanels.clearWindow();
  lastAnalysisAt = 0; chartReset();
}
let rgbRows = [];
function resetResearchValues() {
  researchPanels.reset();
  $('learned-sqi').textContent = '—';
  $('sqi-status').textContent = method === 'deep' ? 'Waiting for 15 seconds of neural pulse samples.' : 'Available in Deep learning mode.';
}
function resetInspector() {
  lastVisualization = null; stateHistory = [[], [], [], [], []];
  clearPanel($('face-roi'), 'Waiting for a face'); clearPanel($('attention-map'), 'Deep learning only');
  clearPanel($('state-trajectory'), 'Deep learning only'); clearPanel($('amplified-video'), 'Amplification off');
}
function resetSession() {
  readings = []; resetResearchValues(); resetSignals(); missingSince = 0; pulseLog = []; previousBox = null; frameTimes = [];
  lastFrameAt = 0; lastFaceAt = 0; lastVideoTime = -1; lastDeepFrameAt = 0;
  ['min', 'average', 'max'].forEach(id => $(id).textContent = '—');
  $('timer').textContent = '00:00'; $('export').disabled = true;
  resetInspector(); sessionMethod = method;
}

document.querySelectorAll('[data-method]').forEach(button => button.addEventListener('click', () => {
  if (running || starting) return;
  method = button.dataset.method;
  document.querySelectorAll('[data-method]').forEach(b => { b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  $('method-description').textContent = method === 'pos'
    ? 'POS projects facial colour changes to reduce shared lighting noise. No pulse model is required.'
    : 'FacePhys is a pretrained state-space neural network. Its model assets load on first use; Python runs inference on this computer.';
  message(method === 'deep' ? 'Deep learning selected. First start downloads the pretrained FacePhys model and runtime.' : 'Classical POS selected. Ready to start.');
  resetResearchValues(); resetInspector();
}));

function updateView() {
  const route = ['monitor', 'benchmark', 'about'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'monitor';
  if (route !== 'monitor' && (running || starting)) stop('Camera stopped while leaving the monitor. Export the session to keep it.');
  document.querySelectorAll('.view').forEach(view => view.hidden = view.id !== route);
  document.querySelectorAll('.nav-link').forEach(link => link.classList.toggle('active', link.hash === '#' + route));
  if (route === 'monitor') renderCharts();
}
window.addEventListener('hashchange', updateView);

async function requestJSON(path, options = {}) {
  let response;
  try { response = await fetch(path, { ...options, cache: 'no-store' }); }
  catch { throw new Error('Python server is not reachable. Keep its Terminal open and run python app.py --port ' + (location.port || '4173') + '.'); }
  let data;
  try { data = await response.json(); }
  catch { throw new Error('Start the Python server with python app.py, then open its localhost address.'); }
  if (!response.ok) throw new Error(data.error || `Python request failed (${response.status}).`);
  return data;
}
function endPythonSession(id) {
  if (id) fetch('/api/stop', { method: 'POST', headers: { 'X-Session-Id': id }, keepalive: true }).catch(() => {});
}
async function start() {
  if (running || starting) return;
  if (offlinePreparing) { message('Wait for the Python model check to finish.'); return; }
  if (!$('consent').checked) { message('Please confirm consent for local webcam estimation before starting.', true); return; }
  if (!navigator.mediaDevices?.getUserMedia) { message('Camera access requires localhost or HTTPS.', true); return; }
  starting = true; const token = ++runId;
  $('start').disabled = true; $('start').textContent = 'Preparing Python…'; $('stop').disabled = false;
  $('prepare-offline').disabled = true; $('check-system').disabled = true;
  document.querySelectorAll('[data-method]').forEach(b => b.disabled = true);
  resetSession();
  try {
    const health = await requestJSON('/api/health');
    if (health.backend !== 'python') throw new Error('This is not the Python server. Run python app.py.');
    message('Python is loading the face tracker and selected pulse model. First use may download weights.');
    const created = await requestJSON('/api/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ method, consent: true }) });
    if (token !== runId) { endPythonSession(created.session); return; }
    pythonSession = created.session;
    message('Waiting for camera permission. Frames will be processed by Python on this Mac.');
    const acquired = await acquireCamera(constraints => navigator.mediaDevices.getUserMedia(constraints),
      { audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: 'user' } }, () => token === runId);
    if (token !== runId) { acquired.getTracks().forEach(t => t.stop()); return; }
    stream = acquired; $('video').srcObject = stream; await $('video').play();
    if (token !== runId) { acquired.getTracks().forEach(t => t.stop()); return; }
    $('overlay').width = $('video').videoWidth; $('overlay').height = $('video').videoHeight;
    $('placeholder').hidden = true; document.querySelector('.camera-stage').classList.add('live');
    $('capture-status').textContent = 'Camera live · Python'; $('capture-status').classList.add('good');
    running = true; starting = false; startedAt = performance.now(); pythonBusy = false;
    $('start').textContent = 'Camera running'; displayInvalid('Finding your face');
    message('Python processing is active. Keep your face evenly lit while the signal builds.');
    frameRequest = requestAnimationFrame(tick);
  } catch (error) {
    if (token === runId) { stop(); message(error.name === 'NotAllowedError' ? 'Camera permission denied. Allow it in this browser and try again.' : error.message, true); }
  }
}
function restoreControls() {
  starting = false; $('start').disabled = false; $('start').textContent = 'Start new session'; $('stop').disabled = true;
  document.querySelectorAll('[data-method]').forEach(b => b.disabled = false);
  $('prepare-offline').disabled = false;
  $('check-system').disabled = false;
}
function stop(text = 'Camera stopped. You can export your session or start a new one.') {
  ++runId; running = false; starting = false; cancelAnimationFrame(frameRequest);
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  $('video').srcObject = null; endPythonSession(pythonSession); pythonSession = null; pythonBusy = false;
  $('placeholder').hidden = false; document.querySelector('.camera-stage').classList.remove('live');
  $('overlay').getContext('2d').clearRect(0, 0, $('overlay').width, $('overlay').height);
  $('capture-status').textContent = 'Camera off'; $('capture-status').classList.remove('good');
  displayInvalid('Session stopped'); restoreControls(); message(text);
  resetInspector(); researchPanels.stop();
}

function append(row) {
  rows.push(row);
  while (rows.length > 0 && row.t - rows[0].t > 60000) rows.shift();
  // One entry per acquired sample; bounded to 30 minutes to avoid unlimited memory.
  if (pulseLog.length < 54000) pulseLog.push({ ...row, v: [...row.v] });
  $('export').disabled = pulseLog.length === 0;
}

const frameCanvas = document.createElement('canvas');
const frameContext = frameCanvas.getContext('2d', { willReadFrequently: true });
function paintPixels(canvas, rgb, size) {
  const rgba = new Uint8ClampedArray(size * size * 4);
  for (let i = 0; i < size * size; i++) { rgba.set(rgb.slice(i * 3, i * 3 + 3), i * 4); rgba[i * 4 + 3] = 255; }
  const buffer = document.createElement('canvas'); buffer.width = buffer.height = size;
  buffer.getContext('2d').putImageData(new ImageData(rgba, size, size), 0, 0);
  canvas.getContext('2d').drawImage(buffer, 0, 0, canvas.width, canvas.height);
}
async function tick(now) {
  if (!running) return;
  frameRequest = requestAnimationFrame(tick);
  const video = $('video');
  if (pythonBusy || video.currentTime === lastVideoTime || now - lastFrameAt < 30 || video.readyState < 2) return;
  const token = runId; pythonBusy = true;
  lastVideoTime = video.currentTime; lastFrameAt = now;
  frameTimes.push(now); while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();
  $('fps').textContent = `${frameTimes.length} FPS`;
  const sec = Math.floor((now - startedAt) / 1000);
  $('timer').textContent = `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`;
  try {
    const scale = Math.min(640 / video.videoWidth, 480 / video.videoHeight, 1);
    frameCanvas.width = Math.round(video.videoWidth * scale); frameCanvas.height = Math.round(video.videoHeight * scale);
    frameContext.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    // Lossless RGBA bytes preserve subtle colour changes; no JPEG compression.
    const pixels = frameContext.getImageData(0, 0, frameCanvas.width, frameCanvas.height).data;
    const data = await requestJSON('/api/frame', { method: 'POST', headers: {
      'Content-Type': 'application/octet-stream', 'X-Session-Id': pythonSession,
      'X-Frame-Width': String(frameCanvas.width), 'X-Frame-Height': String(frameCanvas.height),
      'X-Frame-Time': String(now), 'X-Amplify': String($('amplify').checked), 'X-Gain': $('gain').value
    }, body: pixels });
    if (token !== runId || !running) return;
    const context = $('overlay').getContext('2d'); context.clearRect(0, 0, $('overlay').width, $('overlay').height);
    researchPanels.progress(data.capture);
    if (!data.face) { displayInvalid(data.guidance || 'Face not detected'); $('guidance').textContent = data.guidance; return; }
    const box = data.box; context.strokeStyle = '#bca0dc'; context.lineWidth = 2;
    context.strokeRect(box.x / scale, box.y / scale, box.w / scale, box.h / scale);
    if (data.sample) append(data.sample);
    if (data.crop) paintPixels($('face-roi'), data.crop, 36);
    if (data.amplified) paintPixels($('amplified-video'), data.amplified, 36);
    if (data.visualization) updateInspector(data.visualization);
    if (data.insights) researchPanels.update(data.insights);
    if (Object.hasOwn(data, 'result')) applyResult(data.result);
    $('latency').textContent = `${(data.latency || 0).toFixed(1)} ms`;
    $('capture-details').textContent = `Python ${method === 'pos' ? 'POS' : 'FacePhys'} · ${frameTimes.length} FPS · ${data.capture.pulseSeconds.toFixed(1)} s · ${data.capture.samples} samples`;
    const needed = method === 'deep' ? 450 : 150;
    if (data.capture.samples < needed) displayInvalid(`Collecting signal · ${Math.floor(data.capture.samples / needed * 100)}%`);
  } catch (error) {
    if (token === runId) { stop(); message(error.message, true); }
  } finally { if (token === runId) pythonBusy = false; }
}

function applyResult(result) {
  lastResult = result;
  if (!result) { displayInvalid('No periodic signal yet'); validStreak = 0; return; }
  if (result.learnedSqi != null) {
    $('learned-sqi').textContent = result.learnedSqi.toFixed(2);
    $('sqi-status').textContent = result.learnedSqi > 0.38 ? 'Learned SQI passed. Rate comes from the pretrained spectrum model.' : 'Learned SQI too low; reading hidden.';
  }
  const q = Math.round(result.quality * 100);
  $('quality').textContent = `${q}%`; $('quality-fill').style.width = `${q}%`;
  $('peak-label').textContent = `${result.frequency.toFixed(2)} Hz`;
  $('wave-label').textContent = method === 'pos' ? 'POS · 20 s rolling window' : 'FacePhys · predicted pulse';
  validStreak = result.valid ? validStreak + 1 : 0;
  if (validStreak >= (method === 'deep' ? 1 : 2)) {
    // Median of recent accepted values suppresses isolated fluctuations without
    // manufacturing readings during low-quality intervals.
    rawAccepted.push(result.bpm); if (rawAccepted.length > 3) rawAccepted.shift();
    const recent = [...rawAccepted].sort((a, b) => a - b);
    const bpm = recent[Math.floor(recent.length / 2)];
    const respiratory = result.breathing?.valid ? result.breathing : null;
    const variability = result.variability?.valid ? result.variability : null;
    readings.push({ t: performance.now(), bpm, quality: result.quality, method, sqi: result.learnedSqi,
      breathing: respiratory?.rate, rmssd: variability?.rmssd, sdnn: variability?.sdnn });
    if (readings.length > 1800) readings.shift();
    $('bpm').textContent = bpm.toFixed(0); $('reading-status').textContent = 'Experimental pulse estimate'; $('reading-status').classList.add('good');
    const values = readings.map(r => r.bpm);
    $('min').textContent = Math.min(...values).toFixed(0); $('average').textContent = mean(values).toFixed(0); $('max').textContent = Math.max(...values).toFixed(0);
    $('guidance').textContent = 'Your signal is periodic and stable. Keep the same position for a longer observation.';
    $('app-bpm').value = bpm.toFixed(1); $('benchmark-method').value = method;
  } else {
    displayInvalid(result.valid ? 'Confirming signal…' : 'Signal needs improvement');
    $('guidance').textContent = method === 'deep' ? (result.learnedSqi <= 0.38 ? 'The learned SQI is low. Face the light and keep your face in view.' : 'The model has not produced a plausible pulse estimate yet.') : result.reason;

  }
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
  const output = [['kind', 'seconds', 'method', 'bpm', 'quality', 'red_or_bvp', 'green', 'blue', 'learned_sqi', 'breathing_proxy_per_min', 'prv_rmssd_ms', 'prv_sdnn_ms']];
  for (const row of pulseLog) output.push(['sample', ((row.t - startedAt) / 1000).toFixed(3), sessionMethod, '', '', row.v[0], row.v[1], row.v[2], '', '', '', '']);
  for (const row of readings) output.push(['reading', ((row.t - startedAt) / 1000).toFixed(3), row.method, row.bpm.toFixed(2), row.quality.toFixed(4), '', '', '', row.sqi, row.breathing, row.rmssd, row.sdnn]);
  download(`pulselens-${sessionMethod}-${new Date().toISOString().replaceAll(':', '-')}.csv`, csv(output));
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
window.addEventListener('pagehide', () => { stream?.getTracks().forEach(track => track.stop()); endPythonSession(pythonSession); });
window.addEventListener('resize', renderCharts);
function updateInspector(output) {
  lastVisualization = output;
  ['ssm1', 'ssm2', 'ssm3', 'ssm4', 'proj'].forEach((key, i) => {
    const point = output[key];
    if (point?.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1])) {
      stateHistory[i].push(point.slice(0, 2)); if (stateHistory[i].length > 180) stateHistory[i].shift();
    }
  });
  renderInspector();
}
function renderInspector() {
  if (!lastVisualization) return;
  const index = Number($('model-layer').value);
  heatmap($('attention-map'), lastVisualization['fm' + (Math.min(index, 3) + 1)]);
  trajectory($('state-trajectory'), stateHistory[index]);
}
$('model-layer').addEventListener('change', renderInspector);
$('gain').addEventListener('input', () => $('gain-value').textContent = $('gain').value + '×');
$('amplify').addEventListener('change', () => {
  if (!$('amplify').checked) {
    $('amplified-video').width = $('amplified-video').height = 128;
    clearPanel($('amplified-video'), 'Amplification off');
  }
});

// The Python server must remain running; unregister the old browser-model cache.
if ('serviceWorker' in navigator) navigator.serviceWorker.getRegistrations().then(registrations => {
  const scope = new URL('../', import.meta.url).href;
  registrations.filter(registration => registration.scope === scope).forEach(registration => registration.unregister());
}).catch(() => {});
async function checkPython(button) {
  if (running || starting || offlinePreparing) return;
  offlinePreparing = true; $('check-system').disabled = $('prepare-offline').disabled = true;
  try {
    message('Python is checking both face trackers, 480 recurrent frames, projection, SQI and PSD. No camera is opened.');
    const health = await requestJSON('/api/health');
    if (health.backend !== 'python') throw new Error('Run python app.py to start the Python server.');
    const result = await requestJSON('/api/check', { method: 'POST' });
    if (!result.ok) throw new Error('Python model check did not pass.');
    message(button === 'offline' ? 'Python models are cached locally. Keep this Python server running for offline use.' : 'Python system check passed: face trackers, 480 FacePhys frames, feature projection, SQI and PSD.');
  } catch (error) { message(error.message, true); }
  finally { offlinePreparing = false; $('check-system').disabled = $('prepare-offline').disabled = false; }
}
$('prepare-offline').addEventListener('click', () => checkPython('offline'));
$('check-system').addEventListener('click', () => checkPython('check'));
resetInspector(); resetResearchValues();
updateView();
