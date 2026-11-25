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
test('respiratory trace draws before rate acceptance and survives stop', () => {
  const elements = new Map(); const drawn = [];
  const panels = new ResearchPanels(id => {
    if (!elements.has(id)) elements.set(id, { textContent: '' });
    return elements.get(id);
  }, values => drawn.push([...values]));
  panels.update({ capture: { colorSeconds: 15, pulseSeconds: 15 }, breathing: { valid: false, waveform: [0, 1, 0, -1] } });
  assert.deepEqual(drawn.at(-1), [0, 1, 0, -1]);
  assert.equal(elements.get('breathing-rate').textContent, '—');
  panels.stop(); assert.deepEqual(drawn.at(-1), [0, 1, 0, -1]);
  panels.reset(); assert.deepEqual(drawn.at(-1), []);
});
