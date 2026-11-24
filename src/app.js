import { mean, std, clamp, metrics, csv } from './signal.js';
import { plot } from './charts.js';
import { clearPanel, heatmap, trajectory, TemporalAmplifier } from './visuals.js';
import { FaceRegionTracker, NeuralFaceTracker, NeuralFrameClock, clippedBox } from './tracking.js';
import { checkNeuralRuntime } from './diagnostics.js';
import { acquireCamera } from './camera.js';
import { faceRegions, SkinSampler } from './skin.js';

const $ = id => document.getElementById(id);
let method = 'pos', running = false, starting = false, stream = null, detector = null, landmarkDetector = null;
let analysisWorker = null, deepWorker = null, spectrumWorker = null, frameRequest = 0, runId = 0;
let rows = [], readings = [], pulseLog = [], observations = [];
let previousBox = null, lastVideoTime = -1, lastFrameAt = 0, lastFaceAt = 0;
let lastAnalysisAt = 0, startedAt = 0, deepBusy = false, analysisBusy = false;
let goodFrames = 0, lastResult = null, validStreak = 0, frameTimes = [], rawAccepted = [];
let lastDeepFrameAt = 0;
let captureEpoch = 0, pendingAnalysis = null, learnedSqi = null, lastVisualization = null;
let offlinePreparing = false;
let stateHistory = [[], [], [], [], []], sessionMethod = method;
const amplifier = new TemporalAmplifier();
const skinSampler = new SkinSampler();
const faceTracker = new FaceRegionTracker();
const neuralTracker = new NeuralFaceTracker();
const neuralClock = new NeuralFrameClock();
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
  rgbRows = []; lastDeepFrameAt = 0; amplifier.reset(); resetResearchValues();
  skinSampler.reset(); neuralClock.reset();
  if (deepWorker && running) deepWorker.postMessage({ type: 'reset' });
  lastAnalysisAt = 0; chartReset();
}
let rgbRows = [];
function resetResearchValues() {
  ['breathing-rate', 'hrv-rmssd', 'hrv-sdnn', 'learned-sqi'].forEach(id => $(id).textContent = '—');
  $('breathing-status').textContent = 'Needs at least 30 seconds of steady capture.';
  $('hrv-status').textContent = 'Needs 60 seconds of clean pulse. Exploratory PRV, not ECG-derived HRV.';
  $('beat-count').textContent = 'No accepted beat intervals yet.';
  $('sqi-status').textContent = method === 'deep' ? 'Waiting for 15 seconds of neural pulse samples.' : 'Available in Deep learning mode.';
}
function resetInspector() {
  lastVisualization = null; stateHistory = [[], [], [], [], []];
  clearPanel($('face-roi'), 'Waiting for a face'); clearPanel($('attention-map'), 'Deep learning only');
  clearPanel($('state-trajectory'), 'Deep learning only'); clearPanel($('amplified-video'), 'Amplification off');
}
function resetSession() {
  readings = []; resetSignals(); faceTracker.reset(); neuralTracker.reset(); missingSince = 0; pulseLog = []; previousBox = null; frameTimes = [];
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
    : 'FacePhys is a pretrained state-space neural network. Its model assets load on first use; inference stays on your device.';
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

async function createDetector() {
  const { FaceDetector, FilesetResolver } = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm');
  const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm');
  const modelAssetPath = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite';
  const baseOptions = { modelAssetPath, delegate: 'GPU' };
  const options = { baseOptions, runningMode: 'VIDEO', minDetectionConfidence: 0.5 };
  try { return await FaceDetector.createFromOptions(vision, options); }
  catch { baseOptions.delegate = 'CPU'; return FaceDetector.createFromOptions(vision, options); }
}

async function createLandmarker() {
  const { FaceLandmarker, FilesetResolver } = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm');
  const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm');
  const baseOptions = { modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', delegate: 'GPU' };
  const options = { baseOptions, runningMode: 'VIDEO', numFaces: 2, minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5 };
  try { return await FaceLandmarker.createFromOptions(vision, options); }
  catch { baseOptions.delegate = 'CPU'; return FaceLandmarker.createFromOptions(vision, options); }
}

async function initializeDeep(token) {
  // LiteRT's generated WASM loader calls importScripts: it needs a classic
  // Worker. The worker itself uses dynamic import(), which is supported here.
  const worker = new Worker(new URL('./deep.worker.js', import.meta.url));
  deepWorker = worker;
  spectrumWorker = new Worker(new URL('./spectrum.worker.js', import.meta.url));
  await Promise.all([worker, spectrumWorker].map(current => new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Model loading timed out. Check your connection or local assets.')), 90000);
    current.onerror = event => { clearTimeout(timeout); reject(new Error(event.message || 'Neural worker could not load.')); };
    current.onmessage = ({ data }) => {
      if (data.type === 'progress') message(data.message);
      if (data.type === 'ready' || data.type === 'error') {
        clearTimeout(timeout); data.type === 'ready' ? resolve() : reject(new Error(data.message));
      }
    };
    current.postMessage({ type: 'init' });
  })));
  if (token !== runId) { worker.terminate(); return; }
  worker.onmessage = spectrumWorker.onmessage = ({ data }) => {
    if (token !== runId || !running) return;
    if (data.type === 'error') { stop(); message('Deep learning error: ' + data.message, true); return; }
    if (data.type === 'analysis') {
      if (!pendingAnalysis || data.id !== goodFrames || pendingAnalysis.id !== data.id || data.epoch !== captureEpoch) return;
      const result = pendingAnalysis.result; pendingAnalysis = null;
      learnedSqi = clamp(data.sqi, 0, 1);
      result.learnedSqi = learnedSqi;
      result.valid = learnedSqi > 0.38 && data.bpm >= 40 && data.bpm <= 200 && performance.now() - lastFaceAt < 1500;
      result.bpm = data.bpm;
      const band = data.frequencies.map((frequency, i) => ({ frequency, power: data.powers[i] })).filter(p => p.frequency >= 40 / 60 && p.frequency <= 200 / 60);
      result.frequencies = band.map(p => p.frequency);
      const scale = Math.max(...band.map(p => p.power), 1e-12);
      result.powers = band.map(p => p.power / scale);
      result.quality = learnedSqi;
      result.frequency = data.bpm / 60;
      applyResult(result); return;
    }
    if (data.type === 'pulse') {
      deepBusy = false;
      if (data.epoch !== captureEpoch || performance.now() - lastFaceAt > 1500) return;
      append({ t: data.t, v: [data.value] });
      $('latency').textContent = `${data.latency.toFixed(1)} ms`;
      if (data.visualization) updateInspector(data.visualization);
    }
  };
  worker.onerror = spectrumWorker.onerror = () => { if (token === runId) { stop(); message('Deep inference failed. Restart the camera or select Classical POS.', true); } };
}

async function start() {
  if (running || starting) return;
  if (offlinePreparing) { message('Wait for offline preparation to finish before opening the camera.'); return; }
  if (!$('consent').checked) { message('Please confirm consent for local webcam estimation before starting.', true); return; }
  if (!navigator.mediaDevices?.getUserMedia) { message('Camera access requires localhost or HTTPS in a compatible browser.', true); return; }
  starting = true; const token = ++runId;
  $('start').disabled = true; $('start').textContent = 'Preparing…'; $('stop').disabled = false;
  $('prepare-offline').disabled = true;
  $('check-system').disabled = true;
  document.querySelectorAll('[data-method]').forEach(b => b.disabled = true);
  resetSession();
  message(method === 'deep' ? 'Loading the face detector and pretrained model. First use may take a minute.' : 'Loading the face detector. Your camera will request permission next.');
  try {
    if (method === 'pos' && !landmarkDetector) landmarkDetector = await createLandmarker();
    if (method === 'deep' && !detector) detector = await createDetector();
    if (token !== runId) return;
    if (method === 'deep') await initializeDeep(token);
    if (token !== runId) return;
    message('Waiting for camera permission. Choose Allow for this page in this browser.');
    const acquired = await acquireCamera(constraints => navigator.mediaDevices.getUserMedia(constraints),
      { audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }, facingMode: 'user' } }, () => token === runId);
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
      if (data.id !== goodFrames || performance.now() - lastFaceAt > 1500) return;
      if (method === 'deep' && data.result) {
        if (data.result.modelInput.length !== 450) return;
        pendingAnalysis = data;
        spectrumWorker.postMessage({ type: 'analyze', id: data.id, epoch: captureEpoch, dt: data.result.sampleDt, samples: data.result.modelInput }); return;
      }
      applyResult(data.result);
    };
    analysisWorker.onerror = () => { stop(); message('Signal analysis failed. Please restart the camera.', true); };
    running = true; starting = false; startedAt = performance.now();
    deepBusy = false; analysisBusy = false;
    $('start').textContent = 'Camera running'; displayInvalid('Finding your face');
    message('Keep your face still and evenly lit. Building the first pulse window.');
    frameRequest = requestAnimationFrame(tick);
  } catch (error) {
    if (token === runId) { stop(); message(error.name === 'NotAllowedError' ? 'Camera permission was denied. Allow access in browser settings, then try again.' : error.message, true); }
  } finally { if (token === runId && !running) restoreControls(); }
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
  $('video').srcObject = null; deepWorker?.terminate(); spectrumWorker?.terminate(); analysisWorker?.terminate();
  deepWorker = null; spectrumWorker = null; analysisWorker = null; deepBusy = false; analysisBusy = false;
  $('placeholder').hidden = false; document.querySelector('.camera-stage').classList.remove('live');
  $('overlay').getContext('2d').clearRect(0, 0, $('overlay').width, $('overlay').height);
  $('capture-status').textContent = 'Camera off'; $('capture-status').classList.remove('good');
  displayInvalid('Session stopped'); restoreControls(); message(text);
  resetInspector(); resetResearchValues();
}

function append(row) {
  rows.push(row);
  while (rows.length > 0 && row.t - rows[0].t > 60000) rows.shift();
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
    if (dt > 0.75) resetSignals();
    const landmarks = method === 'pos' ? landmarkDetector.detectForVideo($('video'), now).faceLandmarks : null;
    const regions = landmarks?.map(points => faceRegions(points, $('video').videoWidth, $('video').videoHeight));
    const detected = regions ? regions.map(region => ({ boundingBox: { originX: region.box.x, originY: region.box.y, width: region.box.w, height: region.box.h } })) : detector.detectForVideo($('video'), now).detections;
    const context = $('overlay').getContext('2d'); context.clearRect(0, 0, $('overlay').width, $('overlay').height);
    if (detected.length !== 1) {
      if (!missingSince) missingSince = now;
      if (detected.length > 1 || now - missingSince > 1500) {
        if (rows.length || rgbRows.length) resetSignals();
        previousBox = null; lastFaceAt = 0; resetInspector();
      }
      displayInvalid(detected.length ? 'One person at a time' : 'Face not detected');
      $('guidance').textContent = detected.length ? 'Only one person should be visible in the camera.' : 'Bring your face into view and face the camera.'; return;
    }
    missingSince = 0;
    const b = detected[0].boundingBox;
    const tracked = (method === 'deep' ? neuralTracker : faceTracker).update({ x: b.originX, y: b.originY, w: b.width, h: b.height }, dt);
    const motion = tracked.motion;
    const box = clippedBox(tracked.box, $('video').videoWidth, $('video').videoHeight);
    // Normal head movement changes the crop, but does not erase capture history.
    previousBox = box;
    context.strokeStyle = motion > 0.04 ? '#d6a585' : '#bca0dc'; context.lineWidth = 2;
    context.strokeRect(box.x, box.y, box.w, box.h);
    // Sample three fixed subregions within the detected face, avoiding eyes/mouth.
    if (box.w < 1 || box.h < 1) { displayInvalid('Move your face into the frame'); return; }
    sampleContext.clearRect(0, 0, 64, 64);
    sampleContext.drawImage($('video'), box.x, box.y, box.w, box.h, 0, 0, 64, 64);
    const skin = method === 'pos' ? skinSampler.sample($('video'), regions[0].patches, lastResult?.valid ? lastResult.bpm : null, now) : null;
    const pixels = sampleContext.getImageData(0, 0, 64, 64).data;
    const rgb = skin?.rgb || [0, 1, 2].map(k => { let sum = 0; for (let i = k; i < pixels.length; i += 4) sum += pixels[i]; return sum / (pixels.length / 4); });
    const brightness = mean(rgb);
    if (method === 'pos' && (brightness < 18 || brightness > 245 || box.w < 60 || !skin)) {
      displayInvalid('Adjust distance / lighting');
      $('guidance').textContent = 'Use even lighting and bring your face closer. Your previous session readings are retained.'; return;
    }
    lastFaceAt = now;
    rgbRows.push({ t: now, v: [...rgb] }); while (rgbRows.length && now - rgbRows[0].t > 60000) rgbRows.shift();
    const neuralBox = clippedBox({ x: box.x, y: box.y - box.h * 0.24, w: box.w, h: box.h * 1.2 }, $('video').videoWidth, $('video').videoHeight);
    cropContext.clearRect(0, 0, 36, 36);
    cropContext.drawImage($('video'), neuralBox.x, neuralBox.y, neuralBox.w, neuralBox.h, 0, 0, 36, 36);
    $('face-roi').getContext('2d').drawImage(crop, 0, 0, 180, 180);
    if ($('amplify').checked) {
      const pixels = sampleContext.getImageData(0, 0, 64, 64);
      const enhanced = amplifier.process(pixels, dt, Number($('gain').value));
      const amplified = $('amplified-video');
      amplified.width = 64; amplified.height = 64;
      amplified.getContext('2d').putImageData(new ImageData(enhanced, 64, 64), 0, 0);
    }
    if (method === 'pos') append({ t: now, v: rgb });
    else if (!deepBusy) {
      const rgba = cropContext.getImageData(0, 0, 36, 36).data, pixels = new Float32Array(36 * 36 * 3);
      for (let i = 0; i < 36 * 36; i++) for (let k = 0; k < 3; k++) pixels[i * 3 + k] = rgba[i * 4 + k] / 255;
      const stamp = neuralClock.update(now);
      lastDeepFrameAt = now;
      deepBusy = true; deepWorker.postMessage({ type: 'frame', pixels, dt: stamp.dt, t: stamp.t, epoch: captureEpoch }, [pixels.buffer]);
    }
    const duration = rows.length ? (now - rows[0].t) / 1000 : 0;
    $('capture-details').textContent = `${method === 'pos' ? 'POS' : 'FacePhys'} · ${frameTimes.length} FPS · ${duration.toFixed(1)} s of signal · ${rows.length} samples`;
    const requiredWindow = method === 'deep' ? 450 : 150;
    const ready = rows.length >= requiredWindow;
    if (!ready) {
      displayInvalid(`Collecting signal · ${Math.min(100, Math.floor(rows.length / requiredWindow * 100))}%`);
      $('guidance').textContent = 'Stay still while the signal builds. A longer observation gives better frequency resolution.';
    }
    if (ready && !analysisBusy && !pendingAnalysis && now - lastAnalysisAt > 1000) {
      lastAnalysisAt = now; analysisBusy = true; goodFrames++;
      analysisWorker.postMessage({ method, rows, rgbRows, referenceBpm: lastResult?.valid ? lastResult.bpm : null, id: goodFrames });
    }
  } catch (error) { stop(); message('Capture error: ' + error.message, true); }
}

function applyResult(result) {
  lastResult = result;
  if (!result) { displayInvalid('No periodic signal yet'); validStreak = 0; resetResearchValues(); return; }
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
    $('breathing-rate').textContent = respiratory ? respiratory.rate.toFixed(1) : '—';
    $('breathing-status').textContent = respiratory ? 'Experimental slow colour-variation proxy. Artifacts can look like breathing.' : 'Needs 30 seconds and a clear slow periodic component.';
    $('hrv-rmssd').textContent = variability ? variability.rmssd.toFixed(1) : '—';
    $('hrv-sdnn').textContent = variability ? variability.sdnn.toFixed(1) : '—';
    $('hrv-status').textContent = variability ? 'Exploratory pulse-rate variability; not ECG-derived HRV.' : 'Needs about 60 seconds of clean, consistent beat intervals.';
    $('beat-count').textContent = variability ? `${variability.beatCount} detected beats · ${variability.rejected} rejected intervals` : 'No accepted beat intervals yet.';
    readings.push({ t: performance.now(), bpm, quality: result.quality, method, sqi: result.learnedSqi,
      breathing: respiratory?.rate, rmssd: variability?.rmssd, sdnn: variability?.sdnn });
    if (readings.length > 1800) readings.shift();
    $('bpm').textContent = bpm.toFixed(0); $('reading-status').textContent = 'Stable experimental estimate'; $('reading-status').classList.add('good');
    const values = readings.map(r => r.bpm);
    $('min').textContent = Math.min(...values).toFixed(0); $('average').textContent = mean(values).toFixed(0); $('max').textContent = Math.max(...values).toFixed(0);
    $('guidance').textContent = 'Your signal is periodic and stable. Keep the same position for a longer observation.';
    $('app-bpm').value = bpm.toFixed(1); $('benchmark-method').value = method;
  } else {
    displayInvalid(result.valid ? 'Confirming signal…' : 'Signal needs improvement');
    $('guidance').textContent = method === 'deep' ? (result.learnedSqi <= 0.38 ? 'The learned SQI is low. Face the light and keep your face in view.' : 'The model has not produced a plausible pulse estimate yet.') : result.reason;
    ['breathing-rate', 'hrv-rmssd', 'hrv-sdnn'].forEach(id => $(id).textContent = '—');
    $('breathing-status').textContent = 'Waiting for stable pulse quality.'; $('hrv-status').textContent = 'Waiting for stable pulse quality.';
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
window.addEventListener('pagehide', () => { stream?.getTracks().forEach(track => track.stop()); deepWorker?.terminate(); spectrumWorker?.terminate(); analysisWorker?.terminate(); });
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
  amplifier.reset();
  if (!$('amplify').checked) {
    $('amplified-video').width = $('amplified-video').height = 128;
    clearPanel($('amplified-video'), 'Amplification off');
  }
});

let offlineRegistration = null;
if ('serviceWorker' in navigator && window.isSecureContext) {
  offlineRegistration = navigator.serviceWorker.register(new URL('../sw.js', import.meta.url), { type: 'module', updateViaCache: 'none' });
  offlineRegistration.catch(() => { $('offline-status').textContent = 'Offline cache unavailable'; });
}
$('prepare-offline').addEventListener('click', async () => {
  if (running || starting) return;
  offlinePreparing = true;
  const button = $('prepare-offline'); button.disabled = true;
  try {
    if (!offlineRegistration) throw new Error('Offline support needs HTTPS or localhost and Service Workers.');
    await offlineRegistration; await navigator.serviceWorker.ready;
    if (!navigator.serviceWorker.controller) throw new Error('Reload this page once to activate offline caching, then try again.');
    button.textContent = 'Preparing…'; message('Caching the application, detector, model and runtime. No camera is opened.');
    if (!detector) detector = await createDetector();
    if (!landmarkDetector) landmarkDetector = await createLandmarker();
    await checkNeuralRuntime(progress => message(progress));
    const channel = new MessageChannel();
    const saved = new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Offline cache did not confirm readiness.')), 10000);
      channel.port1.onmessage = ({ data }) => { clearTimeout(timeout); data.ready ? resolve(data) : reject(new Error(data.error || 'Offline assets incomplete.')); };
    });
    navigator.serviceWorker.controller.postMessage({ type: 'prepare' }, [channel.port2]);
    await saved;
    $('offline-status').textContent = '◈ Offline assets prepared';
    message('Offline assets prepared in this browser. Test once with the network disconnected; browser storage can still be cleared.');
  } catch (error) { message(error.message, true); }
  finally { button.textContent = 'Prepare offline use'; button.disabled = false; offlinePreparing = false; }
});
$('check-system').addEventListener('click', async () => {
  if (running || starting || offlinePreparing) return;
  offlinePreparing = true; $('check-system').disabled = true; $('prepare-offline').disabled = true;
  try {
    message('Checking the detector and real neural runtime without opening the camera…');
    if (!detector) detector = await createDetector();
    if (!landmarkDetector) landmarkDetector = await createLandmarker();
    await checkNeuralRuntime(progress => message(progress));
    message('System check passed: face detector, face landmarks, 480 FacePhys frames, feature projection, SQI and PSD all executed. No camera was opened.');
  } catch (error) { message('System check failed: ' + error.message, true); }
  finally { offlinePreparing = false; $('check-system').disabled = false; $('prepare-offline').disabled = false; }
});
resetInspector(); resetResearchValues();
updateView();
