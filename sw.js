// Cache application/model assets only. No webcam frames or measured results
// enter Cache Storage. Changing this version refreshes the source atomically.
const VERSION = 'pulselens-assets-v2';
const ROOT = new URL('./', self.location.href);
const APP = ['index.html', 'favicon.svg', 'manifest.webmanifest', 'THIRD_PARTY_NOTICES.md',
  'src/style.css', 'src/insights.css', 'src/app.js', 'src/signal.js', 'src/vitals.js',
  'src/charts.js', 'src/visuals.js', 'src/analysis.worker.js', 'src/deep.worker.js'];
const MODELS = ['model.tflite', 'state.gz', 'proj.tflite', 'sqi_model.tflite', 'psd_model.tflite'];
const MODEL_ROOT = 'https://raw.githubusercontent.com/xan-antx/pulseLens/e7bdd94cbe686611eb37f817bbdf48b8d770b057/apps/pulselens-deep/models/';
const DETECTOR = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite';

function isAsset(url) {
  if (url.origin === ROOT.origin) return url.href.startsWith(ROOT.href) && (APP.some(path => new URL(path, ROOT).href === url.href) || /\/models\/(model\.tflite|state\.gz|proj\.tflite|sqi_model\.tflite|psd_model\.tflite)$/.test(url.pathname));
  return url.href.startsWith(MODEL_ROOT) || url.href === DETECTOR ||
    (url.hostname === 'cdn.jsdelivr.net' && url.pathname.startsWith('/npm/')) ||
    ['fonts.googleapis.com', 'fonts.gstatic.com'].includes(url.hostname);
}

self.addEventListener('install', event => event.waitUntil((async () => {
  const cache = await caches.open(VERSION); await cache.addAll(APP.map(path => new URL(path, ROOT).href));
  await self.skipWaiting();
})()));
self.addEventListener('activate', event => event.waitUntil((async () => {
  for (const key of await caches.keys()) if (key.startsWith('pulselens-assets-') && key !== VERSION) await caches.delete(key);
  await self.clients.claim();
})()));
self.addEventListener('fetch', event => {
  const request = event.request, url = new URL(request.url);
  if (request.method !== 'GET' || (!isAsset(url) && request.mode !== 'navigate')) return;
  event.respondWith((async () => {
    const cache = await caches.open(VERSION), hit = await cache.match(request);
    if (hit) return hit;
    try {
      const response = await fetch(request);
      if (response.ok && isAsset(url)) await cache.put(request, response.clone());
      return response;
    } catch (error) {
      if (request.mode === 'navigate' && url.origin === ROOT.origin) return (await cache.match(new URL('index.html', ROOT).href)) || Response.error();
      throw error;
    }
  })());
});
self.addEventListener('message', event => {
  if (event.data?.type !== 'prepare') return;
  event.waitUntil((async () => {
    try {
      const cache = await caches.open(VERSION);
      await cache.addAll(APP.map(path => new URL(path, ROOT).href));
      for (const name of MODELS) {
        if (!(await cache.match(MODEL_ROOT + name)) && !(await cache.match(new URL('models/' + name, ROOT).href))) throw new Error('Model cache incomplete: ' + name);
      }
      if (!(await cache.match(DETECTOR))) throw new Error('Face detector cache is incomplete.');
      const keys = (await cache.keys()).map(request => request.url);
      if (!keys.some(key => key.includes('@litertjs/core@0.2.1') && key.endsWith('.wasm')) ||
          !keys.some(key => key.includes('@mediapipe/tasks-vision@0.10.21') && key.endsWith('.wasm'))) throw new Error('WASM runtime cache is incomplete. Reload and retry offline preparation.');
      event.ports[0]?.postMessage({ ready: true });
    } catch (error) { event.ports[0]?.postMessage({ ready: false, error: error.message }); }
  })());
});
