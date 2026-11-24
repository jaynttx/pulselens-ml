import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';
import { isSkin, faceRegions } from '../src/skin.js';

test('learned analysis receives exactly the raw 450 outputs and actual frame interval', async () => {
  let result;
  globalThis.self = { postMessage(data) { result = data; } };
  await import('../src/analysis.worker.js');
  const rows = Array.from({ length: 470 }, (_, i) => ({ t: i * 50, v: [Math.sin(i / 4)] }));
  self.onmessage({ data: { method: 'deep', id: 9, rows, rgbRows: [] } });
  assert.equal(result.id, 9);
  assert.deepEqual(result.result.modelInput, rows.slice(-450).map(r => r.v[0]));
  assert.equal(result.result.sampleDt, 0.05);
  delete globalThis.self;
});
test('constant neural output still reaches learned SQI instead of being blocked by classical estimates', async () => {
  let result;
  globalThis.self = { postMessage(data) { result = data; } };
  // Module was imported above; reload so it installs the handler on this scope.
  await import('../src/analysis.worker.js?constant');
  self.onmessage({ data: { method: 'deep', id: 1, rows: Array.from({ length: 450 }, (_, i) => ({ t: i * 1000 / 30, v: [0] })), rgbRows: [] } });
  assert.equal(result.result.modelInput.length, 450);
  assert.equal(result.result.valid, false);
  delete globalThis.self;
});
test('all neural scripts parse in classic worker mode', async () => {
  for (const name of ['deep.worker.js', 'spectrum.worker.js', 'model-runtime.js']) {
    const code = await readFile(new URL('../src/' + name, import.meta.url), 'utf8');
    assert.doesNotThrow(() => new vm.Script(code));
  }
});
test('face mesh regions stay within video boundaries', () => {
  const points = Array.from({ length: 478 }, (_, i) => ({ x: (i % 20) / 20, y: Math.floor(i / 20) / 24 }));
  const regions = faceRegions(points, 640, 480);
  assert.equal(regions.patches.length, 3);
  for (const box of [regions.box, ...regions.patches]) {
    assert.ok(box.x >= 0 && box.y >= 0 && box.x + box.w <= 640 && box.y + box.h <= 480);
  }
});
test('skin mask excludes black, glare and blue background pixels', () => {
  assert.equal(isSkin(130, 100, 75), true);
  for (const rgb of [[0, 0, 0], [255, 255, 255], [40, 90, 200]]) assert.equal(isSkin(...rgb), false);
});
