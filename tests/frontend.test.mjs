import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// DOM contract smoke tests: no camera or external model is mocked as validated.
// Missing real HTML targets throw instead of silently creating fake elements.
const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
const strokes = new Map();
function element(id = '') {
  const listeners = new Map();
  return { id, value: '', textContent: '', checked: false, disabled: false, hidden: false, style: {}, children: [],
    dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
    addEventListener(type, fn) { listeners.set(type, fn); },
    async dispatch(type, data = {}) { return listeners.get(type)?.({ preventDefault() {}, ...data }); },
    setAttribute() {}, append(...nodes) { this.children.push(...nodes); }, replaceChildren() { this.children = []; },
    getBoundingClientRect() { return { width: 600, height: 120 }; },
    getContext() { return { scale() {}, beginPath() {}, moveTo() {}, lineTo() {}, arc() {}, fill() {},
      stroke() { strokes.set(id, (strokes.get(id) || 0) + 1); }, clearRect() {}, fillRect() {}, fillText() {} }; }
  };
}
const ids = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(match => [match[1], element(match[1])]));
const methods = ['pos', 'deep'].map(method => ({ ...element(), dataset: { method } }));
const views = ['monitor', 'benchmark', 'about'].map(id => ids.get(id));
const nav = ['monitor', 'benchmark', 'about'].map(id => ({ ...element(), hash: '#' + id }));
const events = new Map();
globalThis.document = { hidden: false,
  getElementById(id) { assert.ok(ids.has(id), 'Missing HTML element: ' + id); return ids.get(id); },
  querySelectorAll(selector) { return selector === '[data-method]' ? methods : selector === '.view' ? views : selector === '.nav-link' ? nav : []; },
  querySelector() { return element(); }, createElement() { return element(); },
  addEventListener(type, fn) { events.set(type, fn); }
};
globalThis.window = { isSecureContext: false, addEventListener(type, fn) { events.set(type, fn); } };
globalThis.location = { hash: '#monitor' };
const { applyResult } = await import('../src/app.js');

test('frontend initializes every real advanced panel', () => {
  assert.equal(ids.get('monitor').hidden, false);
  assert.equal(ids.get('benchmark').hidden, true);
  for (const id of ['breathing-rate', 'hrv-rmssd', 'hrv-sdnn', 'learned-sqi']) assert.equal(ids.get(id).textContent, '—');
});
test('method selector updates frontend and model information', async () => {
  await methods[1].dispatch('click');
  assert.match(ids.get('method-description').textContent, /pretrained/);
  assert.match(ids.get('sqi-status').textContent, /15 seconds/);
  await methods[0].dispatch('click');
  assert.match(ids.get('method-description').textContent, /POS/);
});
test('camera stays off until the local consent box is checked', async () => {
  await ids.get('start').dispatch('click');
  assert.match(ids.get('notice').textContent, /confirm consent/);
});
test('benchmark form updates calculated frontend metrics', async () => {
  ids.get('app-bpm').value = '74'; ids.get('ref-bpm').value = '70';
  ids.get('condition').value = 'Good light'; ids.get('benchmark-method').value = 'pos';
  await ids.get('benchmark-form').dispatch('submit');
  assert.match(ids.get('benchmark-metrics').textContent, /MAE 4.00 BPM/);
  assert.equal(ids.get('observations').children.length, 1);
  assert.equal(ids.get('export-benchmark').disabled, false);
});
test('amplification slider updates its visible gain label', async () => {
  ids.get('gain').value = '18'; await ids.get('gain').dispatch('input');
  assert.equal(ids.get('gain-value').textContent, '18×');
});
test('navigation opens the requested frontend view', () => {
  location.hash = '#about'; events.get('hashchange')();
  assert.equal(ids.get('about').hidden, false); assert.equal(ids.get('monitor').hidden, true);
});
test('accepted pulse estimates draw a visible trend in both methods', async () => {
  const result = { valid: true, bpm: 77, frequency: 77 / 60, quality: .9, waveform: [0, 1, 0], powers: [0, 1, 0], windowSeconds: 20 };
  for (const button of methods) {
    await button.dispatch('click');
    const before = strokes.get('trend') || 0;
    applyResult({ ...result, learnedSqi: button.dataset.method === 'deep' ? .9 : undefined });
    applyResult({ ...result, learnedSqi: button.dataset.method === 'deep' ? .9 : undefined });
    assert.equal(ids.get('bpm').textContent, '77');
    assert.match(ids.get('trend-status').textContent, /accepted readings/);
    assert.ok(strokes.get('trend') > before, 'Trend must redraw after an accepted estimate');
  }
});
