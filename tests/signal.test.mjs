import test from 'node:test';
import assert from 'node:assert/strict';
import { pos, estimate, resample, detrend, metrics, csv } from '../src/signal.js';

for (const bpm of [48, 60, 72, 90, 120, 150, 180]) {
  test(`spectral estimate recovers ${bpm} BPM`, () => {
    const wave = Array.from({ length: 600 }, (_, i) => Math.sin(2 * Math.PI * bpm / 60 * i / 30) + 0.2 * Math.sin(2 * Math.PI * 0.1 * i / 30));
    const result = estimate(wave);
    assert.ok(Math.abs(result.bpm - bpm) < 1, JSON.stringify(result.bpm));
    assert.equal(result.valid, true);
  });
}
test('POS recovers a pulse in colour traces with shared lighting drift', () => {
  const rgb = Array.from({ length: 600 }, (_, i) => {
    const pulse = Math.sin(2 * Math.PI * 1.2 * i / 30), light = 1 + 0.03 * Math.sin(2 * Math.PI * 0.12 * i / 30);
    return [150 * light + 0.1 * pulse, 105 * light + 0.8 * pulse, 85 * light + 0.15 * pulse];
  });
  const result = estimate(pos(rgb)); assert.ok(Math.abs(result.bpm - 72) < 1); assert.ok(result.valid);
});
test('no estimate for flat, non-finite, or insufficient input', () => {
  assert.equal(estimate(Array(600).fill(0)), null);
  assert.equal(estimate(Array(600).fill(NaN)), null);
  assert.equal(estimate(Array(100).fill(1)), null);
});
test('uniform resampling uses elapsed timestamps', () => {
  const data = resample([{ t: 0, v: [0] }, { t: 400, v: [4] }, { t: 1000, v: [10] }], 2);
  assert.deepEqual(data.map(r => r[0]), [0, 5, 10]);
});
test('linear drift is removed', () => { assert.ok(detrend([2, 4, 6, 8, 10]).every(x => Math.abs(x) < 1e-10)); });
test('paired metrics use each observations own reference', () => {
  const m = metrics([{ app: 74, reference: 70 }, { app: 78, reference: 80 }]);
  assert.equal(m.mae, 3); assert.equal(m.bias, 1); assert.equal(m.rmse, Math.sqrt(10));
});
test('CSV protects string formulas and escapes quotes', () => {
  assert.equal(csv([['=1+1', 'a"b', -2]]), '"\'=1+1","a""b","-2"');
});
test('broadband noise fails quality gating', () => {
  let seed = 42; const random = () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };
  assert.equal(estimate(Array.from({ length: 600 }, random)).valid, false);
});
