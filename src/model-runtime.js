// Shared classic-worker loader. LiteRT's WASM bootstrap needs importScripts.
self.PulseModels = (async () => {
  const base = 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/';
  const remote = 'https://raw.githubusercontent.com/xan-antx/pulseLens/e7bdd94cbe686611eb37f817bbdf48b8d770b057/apps/pulselens-deep/models/';
  const api = await import(base + '+esm');
  const fetchOriginal = self.fetch;
  self.fetch = (url, init) => fetchOriginal(typeof url === 'string' && url.endsWith('.wasm') ? base + 'wasm/' + url.split('/').pop() : url, init);
  try { await api.loadLiteRt(base + 'wasm/'); }
  finally { self.fetch = fetchOriginal; }
  async function asset(name) {
    const local = await fetch(new URL('../models/' + name, self.location.href)).catch(() => null);
    if (local?.ok) return local;
    const response = await fetch(remote + name);
    if (!response.ok) throw new Error('Unable to load model asset: ' + name);
    return response;
  }
  async function compile(name) {
    const url = URL.createObjectURL(new Blob([await (await asset(name)).arrayBuffer()]));
    try { return await api.loadAndCompile(url, { accelerator: 'wasm' }); }
    finally { URL.revokeObjectURL(url); }
  }
  return { api, asset, compile };
})();
