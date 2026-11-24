import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';

const code = await readFile(new URL('../sw.js', import.meta.url), 'utf8');
function environment() {
  const events = {}, stores = new Map(), network = [];
  const storage = { async keys() { return [...stores.keys()]; }, async delete(name) { return stores.delete(name); },
    async open(name) {
      if (!stores.has(name)) stores.set(name, new Map());
      const map = stores.get(name), key = request => typeof request === 'string' ? request : request.url;
      return { async addAll(urls) { urls.forEach(url => map.set(url, new Response(url))); },
        async match(request) { return map.get(key(request))?.clone(); },
        async put(request, response) { map.set(key(request), response); },
        async keys() { return [...map.keys()].map(url => new Request(url)); } };
    }
  };
  const context = { URL, Response, caches: storage,
    self: { location: { href: 'https://example.test/pulse/sw.js' }, addEventListener(name, fn) { events[name] = fn; },
      async skipWaiting() {}, clients: { async claim() {} } },
    async fetch(request) { network.push(request); throw new Error('offline'); }
  };
  vm.runInNewContext(code, context);
  async function dispatch(name, input = {}) { let pending; events[name]({ ...input, waitUntil(promise) { pending = promise; } }); await pending; }
  async function fetchPage(url, mode = 'cors', method = 'GET') {
    let response; events.fetch({ request: { url, method, mode }, respondWith(promise) { response = promise; } }); return response;
  }
  return { storage, network, dispatch, fetchPage, stores };
}
test('offline navigation falls back to cached application shell', async () => {
  const env = environment(); await env.dispatch('install');
  const response = await env.fetchPage('https://example.test/pulse/', 'navigate');
  assert.equal(await response.text(), 'https://example.test/pulse/index.html');
});
test('cached application modules load without any network request', async () => {
  const env = environment(); await env.dispatch('install');
  const response = await env.fetchPage('https://example.test/pulse/src/vitals.js');
  assert.ok(response.ok); assert.equal(env.network.length, 0);
});
test('physiological POSTs and unlisted API paths are not cached or intercepted', async () => {
  const env = environment(); await env.dispatch('install');
  assert.equal(await env.fetchPage('https://example.test/pulse/api/measurements', 'cors', 'POST'), undefined);
  assert.equal(await env.fetchPage('https://example.test/pulse/api/measurements'), undefined);
  assert.equal(env.network.length, 0);
});
test('offline preparation refuses incomplete model/runtime caches', async () => {
  const env = environment(); let reply;
  await env.dispatch('message', { data: { type: 'prepare' }, ports: [{ postMessage(data) { reply = data; } }] });
  assert.equal(reply.ready, false); assert.match(reply.error, /Model cache incomplete/);
});
test('cache activation removes old PulseLens versions and preserves other apps', async () => {
  const env = environment(); await env.storage.open('pulselens-assets-v1'); await env.storage.open('other-app');
  await env.dispatch('install'); await env.dispatch('activate');
  assert.ok(!env.stores.has('pulselens-assets-v1')); assert.ok(env.stores.has('other-app'));
});
