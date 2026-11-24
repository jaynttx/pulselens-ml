// Face-mesh topology is model data; extraction and weighting are implemented here.
import { clippedBox } from './tracking.js';
const PATCHES = [[10, 67, 69, 104, 108, 151, 284, 298, 299, 337], [123, 147, 187, 205, 206, 216], [352, 376, 411, 425, 426, 436]];
function bounds(points, width, height, pad = 0) {
  const xs = points.map(p => p.x * width), ys = points.map(p => p.y * height);
  const x = Math.min(...xs) - pad, y = Math.min(...ys) - pad;
  return clippedBox({ x, y, w: Math.max(...xs) + pad - x, h: Math.max(...ys) + pad - y }, width, height);
}
export function faceRegions(points, width, height) {
  return { box: bounds(points, width, height), patches: PATCHES.map(indices => bounds(indices.map(i => points[i]), width, height, 4)) };
}
export function isSkin(r, g, b) {
  const total = r + g + b;
  return total >= 60 && r / total > 0.35 && r / total < 0.6 && g / total > 0.25 && g / total < 0.4;
}
// Learn a spatial mask from how strongly each location follows the current
// pulse frequency. Each region has its own history, so cheek weights cannot
// accidentally be learned from forehead pixels.
export class SkinSampler {
  constructor(size = 16) { this.size = size; this.reset(); }
  reset() { this.history = [[], [], []]; this.weights = [[], [], []]; this.frames = 0; }
  sample(video, regions, bpm, now) {
    this.canvas ||= document.createElement('canvas');
    this.canvas.width = this.canvas.height = this.size;
    const context = this.canvas.getContext('2d', { willReadFrequently: true });
    const sum = [0, 0, 0]; let mass = 0;
    regions.forEach((region, regionIndex) => {
      if (region.w < 1 || region.h < 1) return;
      context.drawImage(video, region.x, region.y, region.w, region.h, 0, 0, this.size, this.size);
      const rgba = context.getImageData(0, 0, this.size, this.size).data;
      const trace = new Float32Array(this.size ** 2);
      for (let pixel = 0; pixel < trace.length; pixel++) {
        const i = pixel * 4, [r, g, b] = rgba.subarray(i, i + 3);
        trace[pixel] = g / Math.max(1, r + g + b);
        if (!isSkin(r, g, b)) continue;
        const weight = this.weights[regionIndex][pixel] ?? 1;
        sum[0] += r * weight; sum[1] += g * weight; sum[2] += b * weight; mass += weight;
      }
      const history = this.history[regionIndex]; history.push({ t: now, trace });
      if (history.length > 64) history.shift();
      if (bpm && history.length === 64 && this.frames % 15 === 0) {
        this.weights[regionIndex] = trace.map((_, pixel) => {
          const average = history.reduce((s, row) => s + row.trace[pixel], 0) / history.length;
          let real = 0, imaginary = 0, energy = 0;
          for (const row of history) {
            const v = row.trace[pixel] - average;
            const phase = 2 * Math.PI * bpm / 60 * (row.t - history[0].t) / 1000;
            real += v * Math.cos(phase); imaginary += v * Math.sin(phase); energy += v * v;
          }
          const coherence = Math.min(1, 2 * (real * real + imaginary * imaginary) / Math.max(1e-12, history.length * energy));
          return 0.1 + 0.9 * coherence;
        });
      }
    });
    this.frames++;
    return mass > 0 ? { rgb: sum.map(v => v / mass), skinWeight: mass } : null;
  }
}
