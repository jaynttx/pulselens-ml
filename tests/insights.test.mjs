import test from 'node:test';
import assert from 'node:assert/strict';
import { ResearchPanels } from '../src/insights.js';

function panel() {
  const elements = new Map();
  const find = id => { if (!elements.has(id)) elements.set(id, { textContent: '' }); return elements.get(id); };
  const panels = new ResearchPanels(find); panels.reset();
  return { panels, text: id => find(id).textContent };
}
const sample = { capture: { colorSeconds: 60, pulseSeconds: 60 },
  breathing: { valid: true, rate: 18 }, variability: { valid: true, rmssd: 12.2, sdnn: 15.1, beatCount: 71, rejected: 2 } };

test('research values render independently of the current BPM gate', () => {
  const { panels, text } = panel();
  panels.update({ ...sample, valid: false });
  assert.equal(text('breathing-rate'), '18.0');
  assert.equal(text('hrv-rmssd'), '12.2');
  assert.equal(text('hrv-sdnn'), '15.1');
});
test('each panel reports its actual captured window', () => {
  const { panels, text } = panel();
  panels.progress({ colorSeconds: 24.2, pulseSeconds: 18.2 });
  assert.match(text('breathing-status'), /24 \/ 30/);
  assert.match(text('hrv-status'), /18 \/ 60/);
  panels.progress({ colorSeconds: 60, pulseSeconds: 59.97 });
  assert.match(text('hrv-status'), /waiting for consistent beat/);
});
test('stopping preserves completed values and labels them as historical', () => {
  const { panels, text } = panel(); panels.update(sample); panels.stop();
  assert.equal(text('breathing-rate'), '18.0'); assert.equal(text('hrv-rmssd'), '12.2');
  assert.match(text('breathing-status'), /Session stopped/); assert.match(text('hrv-status'), /Session stopped/);
});
test('quality loss labels previous values rather than claiming a fresh estimate', () => {
  const { panels, text } = panel(); panels.update(sample);
  panels.update({ capture: sample.capture, breathing: null, variability: null });
  assert.equal(text('breathing-rate'), '18.0'); assert.match(text('breathing-status'), /Last accepted window/);
  panels.reset(); assert.equal(text('breathing-rate'), '—'); assert.equal(text('hrv-rmssd'), '—');
});
test('worker computes respiration even when flat pulse produces no BPM result', async () => {
  let response;
  globalThis.self = { postMessage(data) { response = data; } };
  await import('../src/analysis.worker.js?independent-insights');
  const rows = Array.from({ length: 1801 }, (_, i) => ({ t: i * 1000 / 30, v: [100, 100, 100] }));
  const rgbRows = rows.map((row, i) => ({ t: row.t, v: [130, 100 + 2 * Math.sin(2 * Math.PI * 0.3 * i / 30), 75] }));
  self.onmessage({ data: { method: 'pos', rows, rgbRows, id: 7 } });
  assert.equal(response.result, null);
  assert.ok(response.insights.breathing.valid);
  assert.ok(Math.abs(response.insights.breathing.rate - 18) < 0.5);
  assert.equal(response.insights.capture.pulseSeconds, 60);
  delete globalThis.self;
});
