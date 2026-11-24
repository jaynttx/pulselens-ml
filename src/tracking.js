// Smooth the sampled regions, rather than letting detector-box jitter change
// the mixture of skin/background pixels on every frame.
export class FaceRegionTracker {
  constructor() { this.reset(); }
  reset() { this.raw = null; this.box = null; }
  update(raw, dt = 1 / 30) {
    const motion = this.raw ? Math.hypot(raw.x - this.raw.x, raw.y - this.raw.y) / raw.w + Math.abs(raw.w - this.raw.w) / raw.w : 0;
    const changed = motion > 0.35;
    this.raw = { ...raw };
    if (!this.box || changed) this.box = { ...raw };
    else {
      const position = 1 - Math.exp(-dt / 0.18), size = 1 - Math.exp(-dt / 0.6);
      for (const key of ['x', 'y']) this.box[key] += position * (raw[key] - this.box[key]);
      for (const key of ['w', 'h']) this.box[key] += size * (raw[key] - this.box[key]);
    }
    return { box: { ...this.box }, motion, changed };
  }
}

export function clippedBox(box, width, height) {
  const x = Math.max(0, Math.min(width, box.x)), y = Math.max(0, Math.min(height, box.y));
  return { x, y, w: Math.max(0, Math.min(width, box.x + box.w) - x), h: Math.max(0, Math.min(height, box.y + box.h) - y) };
}

// Independent implementation of scalar Kalman filtering for the neural crop.
// Use the reference pipeline's Q=.01, R=.5 and initial covariance=1.
export class NeuralFaceTracker {
  constructor() { this.reset(); }
  reset() { this.values = null; this.variance = [1, 1, 1, 1]; this.previous = null; }
  update(box) {
    const keys = ['x', 'y', 'w', 'h'];
    const observations = keys.map(key => box[key]);
    const motion = this.previous ? Math.hypot(box.x - this.previous.x, box.y - this.previous.y) / Math.max(1, box.w) : 0;
    if (!this.values) this.values = observations;
    else this.values = observations.map((observation, i) => {
      const predicted = this.variance[i] + 0.01;
      const gain = predicted / (predicted + 0.5);
      this.variance[i] = (1 - gain) * predicted;
      return this.values[i] + gain * (observation - this.values[i]);
    });
    this.previous = { ...box };
    return { box: Object.fromEntries(keys.map((key, i) => [key, this.values[i]])), motion };
  }
}

export class NeuralFrameClock {
  constructor() { this.reset(); }
  reset() { this.last = null; this.time = null; this.dt = 1 / 30; }
  update(now) {
    if (!Number.isFinite(now) || (this.last !== null && now <= this.last)) throw new Error('Neural frame timestamps must increase.');
    if (this.last === null) this.time = now;
    else {
      this.dt += 0.003 * ((now - this.last) / 1000 - this.dt);
      this.time += this.dt * 1000;
      this.time += 0.003 * (now - this.time);
    }
    this.last = now;
    return { dt: this.dt, t: this.time };
  }
}
