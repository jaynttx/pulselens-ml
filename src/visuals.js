export function clearPanel(canvas, text) {
  const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#f5f0fa'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#8a749b'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
}

export function heatmap(canvas, data) {
  const side = Math.sqrt(data?.length || 0);
  if (!Number.isInteger(side) || side < 2 || data.some(v => !Number.isFinite(v))) return clearPanel(canvas, 'No feature map');
  const lo = Math.min(...data), hi = Math.max(...data), range = hi - lo || 1;
  const small = document.createElement('canvas'); small.width = small.height = side;
  const ctx = small.getContext('2d'), pixels = ctx.createImageData(side, side);
  data.forEach((value, i) => {
    const t = (value - lo) / range;
    pixels.data.set([245 - t * 90, 232 - t * 140, 255 - t * 45, 255], i * 4);
  });
  ctx.putImageData(pixels, 0, 0);
  const destination = canvas.getContext('2d'); destination.imageSmoothingEnabled = false;
  destination.drawImage(small, 0, 0, canvas.width, canvas.height);
}

export function trajectory(canvas, points) {
  if (points.length < 3) return clearPanel(canvas, 'Waiting for model states');
  const ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#f8f4fc'; ctx.fillRect(0, 0, w, h);
  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const minX = Math.min(...xs), minY = Math.min(...ys), dx = Math.max(...xs) - minX || 1, dy = Math.max(...ys) - minY || 1;
  for (let i = 1; i < points.length; i++) {
    ctx.beginPath(); ctx.moveTo(12 + (points[i - 1][0] - minX) / dx * (w - 24), h - 12 - (points[i - 1][1] - minY) / dy * (h - 24));
    ctx.lineTo(12 + (points[i][0] - minX) / dx * (w - 24), h - 12 - (points[i][1] - minY) / dy * (h - 24));
    ctx.strokeStyle = `rgba(128,86,172,${0.1 + i / points.length * 0.9})`; ctx.lineWidth = 2; ctx.stroke();
  }
}

// True pixel-wise temporal amplification, not a decorative BPM animation.
export class TemporalAmplifier {
  constructor() { this.reset(); }
  reset() { this.fast = null; this.slow = null; }
  process(pixels, dt, gain) {
    const data = pixels.data;
    if (!this.fast) { this.fast = Float32Array.from(data); this.slow = Float32Array.from(data); }
    const a = 1 - Math.exp(-2 * Math.PI * 3.33 * Math.min(dt, 0.1));
    const b = 1 - Math.exp(-2 * Math.PI * 0.67 * Math.min(dt, 0.1));
    const output = new Uint8ClampedArray(data.length);
    for (let i = 0; i < data.length; i++) {
      if (i % 4 === 3) { output[i] = 255; continue; }
      this.fast[i] += a * (data[i] - this.fast[i]); this.slow[i] += b * (data[i] - this.slow[i]);
      output[i] = data[i] + gain * (this.fast[i] - this.slow[i]);
    }
    return output;
  }
}
