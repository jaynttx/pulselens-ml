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
