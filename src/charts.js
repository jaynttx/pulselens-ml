export function plot(canvas, values = [], { min, max, bars = false, color = '#aa8bc8' } = {}) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  const w = rect.width, h = rect.height, padding = 12;
  ctx.strokeStyle = '#f0ecf5'; ctx.lineWidth = 1;
  for (let j = 0; j < 4; j++) { const y = padding + (h - 2 * padding) * j / 3; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  if (values.length === 0) {
    ctx.fillStyle = '#b5a7be'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Your signal will appear here', w / 2, h / 2); return;
  }
  if (values.length === 1) {
    ctx.fillStyle = color; ctx.beginPath();
    ctx.arc(w / 2, h / 2, 4, 0, 2 * Math.PI); ctx.fill(); return;
  }
  const lower = min ?? Math.min(...values), upper = max ?? Math.max(...values);
  const height = Math.max(upper - lower, 0.001);
  const point = (v, i) => [i / (values.length - 1) * w, h - padding - (v - lower) / height * (h - padding * 2)];
  if (bars) {
    ctx.fillStyle = color;
    values.forEach((v, i) => { const [x, y] = point(v, i); ctx.fillRect(x, y, Math.max(1, w / values.length - 1), h - padding - y); }); return;
  }
  ctx.beginPath(); values.forEach((v, i) => { const [x, y] = point(v, i); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  const gradient = ctx.createLinearGradient(0, 0, 0, h); gradient.addColorStop(0, '#b393cf28'); gradient.addColorStop(1, '#b393cf00');
  ctx.fillStyle = gradient; ctx.fill();
}
