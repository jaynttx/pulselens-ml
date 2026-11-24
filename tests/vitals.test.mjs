import test from 'node:test';
import assert from 'node:assert/strict';
import { breathing, variability, bandpass } from '../src/vitals.js';
import { TemporalAmplifier } from '../src/visuals.js';

for (const rate of [12, 18, 24]) {
  test(`respiratory proxy recovers ${rate} cycles per minute`, () => {
    const rows = Array.from({ length: 1801 }, (_, i) => ({ t: i * 1000 / 30,
      v: [150, 100 + 2 * Math.sin(2 * Math.PI * rate / 60 * i / 30) + 0.2 * Math.sin(2 * Math.PI * 1.2 * i / 30), 80] }));
    const result = breathing(rows);
    assert.ok(result.valid); assert.ok(Math.abs(result.rate - rate) <= 0.5);
  });
}
test('breathing is unavailable for short or flat input', () => {
  assert.equal(breathing([{ t: 0, v: [1, 1, 1] }, { t: 29000, v: [1, 1, 1] }]), null);
  assert.equal(breathing([{ t: 0, v: [1, 1, 1] }, { t: 60000, v: [1, 1, 1] }]), null);
});
test('regular synthetic pulse has near-zero variability', () => {
  const signal = Array.from({ length: 1800 }, (_, i) => Math.sin(2 * Math.PI * 1.25 * i / 30));
  const result = variability(signal, 30, 75);
  assert.ok(result?.valid); assert.ok(result.rmssd < 3); assert.ok(result.sdnn < 3);
  assert.ok(result.beatCount >= 65);
});
test('variable synthetic beat intervals produce nonzero RMSSD', () => {
  const beats = []; let time = 0;
  for (let i = 0; time < 61; i++) { beats.push(time); time += 0.8 + 0.045 * Math.sin(i * 0.6); }
  const signal = Array.from({ length: 1800 }, (_, i) => beats.reduce((sum, beat) => sum + Math.exp(-(((i / 30 - beat) / 0.07) ** 2)), 0));
  const result = variability(signal, 30, 75);
  assert.ok(result?.valid); assert.ok(result.rmssd > 10 && result.rmssd < 40); assert.ok(result.sdnn > 20);
});
test('variability requires a long pulse window and expected BPM', () => {
  assert.equal(variability(Array(1000).fill(0), 30, 75), null);
  assert.equal(variability(Array(1800).fill(1), 30, 75), null);
  assert.equal(variability(Array(1800).fill(1), 30, NaN), null);
});
test('temporal amplification never adds variation to a constant image', () => {
  const amp = new TemporalAmplifier(), frame = { data: new Uint8ClampedArray([120, 100, 80, 255]) };
  for (let i = 0; i < 100; i++) assert.deepEqual(Array.from(amp.process(frame, 1 / 30, 20)), [120, 100, 80, 255]);
});
test('temporal amplification increases an actual periodic pixel change', () => {
  const amp = new TemporalAmplifier(), output = [], input = [];
  for (let i = 0; i < 300; i++) {
    const green = Math.round(100 + 3 * Math.sin(2 * Math.PI * 1.2 * i / 30));
    const data = amp.process({ data: new Uint8ClampedArray([120, green, 80, 255]) }, 1 / 30, 10);
    if (i > 100) { input.push(green); output.push(data[1]); }
  }
  assert.ok(Math.max(...output) - Math.min(...output) > 3 * (Math.max(...input) - Math.min(...input)));
  amp.reset(); assert.equal(amp.fast, null);
});
test('bandpass stays finite for constant and impulse data', () => {
  assert.ok(bandpass(Array(300).fill(100)).every(v => v === 0));
  assert.ok(bandpass([1, ...Array(299).fill(0)]).every(Number.isFinite));
});
