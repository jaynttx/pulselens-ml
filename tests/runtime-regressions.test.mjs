import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFile } from 'node:fs/promises';
import { estimate } from '../src/signal.js';
import { FaceRegionTracker, clippedBox } from '../src/tracking.js';
import { checkNeuralRuntime } from '../src/diagnostics.js';
import { acquireCamera } from '../src/camera.js';

test('LiteRT worker parses as a classic worker (no import.meta/module syntax)', async () => {
  const code = await readFile(new URL('../src/deep.worker.js', import.meta.url), 'utf8');
  assert.doesNotThrow(() => new vm.Script(code));
});
test('quality checking removes large out-of-band illumination drift', () => {
  const pulse = Array.from({ length: 600 }, (_, i) => Math.sin(2 * Math.PI * 1.2 * i / 30) + 20 * Math.sin(2 * Math.PI * 0.12 * i / 30));
  const result = estimate(pulse);
  assert.ok(result.valid, JSON.stringify({ quality: result.quality, reason: result.reason }));
  assert.ok(Math.abs(result.bpm - 72) < 1);
});
test('face-box smoothing reduces pixel-region jitter without resetting tracking', () => {
  const tracker = new FaceRegionTracker(), centers = [];
  for (let i = 0; i < 100; i++) {
    const result = tracker.update({ x: 100 + (i % 2 ? 3 : -3), y: 80, w: 150 + (i % 2 ? 2 : -2), h: 180 });
    if (i > 30) centers.push(result.box.x);
    assert.equal(result.changed, false);
  }
  assert.ok(Math.max(...centers) - Math.min(...centers) < 1);
});
test('cropping clamps face rectangles at all video boundaries', () => {
  assert.deepEqual(clippedBox({ x: -20, y: -10, w: 150, h: 170 }, 640, 480), { x: 0, y: 0, w: 130, h: 160 });
  assert.deepEqual(clippedBox({ x: 620, y: 460, w: 100, h: 100 }, 640, 480), { x: 620, y: 460, w: 20, h: 20 });
});
test('system check uses classic workers and exercises inference before reporting success', async () => {
  let calls = 0, terminated = false;
  class FakeWorker {
    constructor(url, options) { assert.match(url.href, /(deep|spectrum)\.worker\.js$/); assert.equal(options, undefined); }
    postMessage(data) {
      queueMicrotask(() => {
        if (data.type === 'init') this.onmessage({ data: { type: 'ready' } });
        if (data.type === 'frame') { calls++; this.onmessage({ data: { type: 'pulse', value: 0.01, visualization: calls % 5 === 0 ? { fm1: [1, 2, 3, 4] } : null } }); }
        if (data.type === 'analyze') { assert.ok(calls >= 6); assert.equal(data.samples.length, 450); assert.ok(data.samples.every(Number.isFinite)); this.onmessage({ data: { type: 'analysis', sqi: 0.8, bpm: 72 } }); }
      });
    }
    terminate() { terminated = true; }
  }
  const result = await checkNeuralRuntime(() => {}, FakeWorker);
  assert.ok(calls >= 6); assert.equal(result.projection, true); assert.equal(terminated, true);
});
test('system check surfaces actual worker startup errors', async () => {
  class FailingWorker {
    postMessage() { queueMicrotask(() => this.onmessage({ data: { type: 'error', message: 'importScripts is unavailable' } })); }
    terminate() {}
  }
  await assert.rejects(checkNeuralRuntime(() => {}, FailingWorker), /importScripts/);
});
test('an unanswered permission request times out with an actionable error', async () => {
  await assert.rejects(acquireCamera(() => new Promise(() => {}), {}, () => true, 5), /permission is still pending/);
});
test('a stream arriving after cancellation is immediately released', async () => {
  let stopped = false;
  await assert.rejects(acquireCamera(async () => ({ getTracks: () => [{ stop() { stopped = true; } }] }), {}, () => false), /cancelled/);
  assert.equal(stopped, true);
});
