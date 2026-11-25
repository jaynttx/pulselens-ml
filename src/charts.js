export function plot(canvas, values = [], { min, max, bars = false, color = '#aa8bc8', empty = 'Your signal will appear here', markers = false, times } = {}) {
  values = values.map(value => Number.isFinite(value) ? value : null);
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio);
  const w = rect.width, h = rect.height, padding = 12;
  ctx.strokeStyle = '#f0ecf5'; ctx.lineWidth = 1;
  for (let j = 0; j < 4; j++) { const y = padding + (h - 2 * padding) * j / 3; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  if (!values.some(value => value !== null)) {
    ctx.fillStyle = '#b5a7be'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(empty, w / 2, h / 2); return;
  }
  if (values.length === 1) {
    ctx.fillStyle = color; ctx.beginPath();
    ctx.arc(w / 2, h / 2, 4, 0, 2 * Math.PI); ctx.fill(); return;
  }
  const finite = values.filter(value => value !== null);
  const lower = min ?? Math.min(...finite) - .1, upper = max ?? Math.max(...finite) + .1;
  const height = Math.max(upper - lower, 0.001);
  const duration = times?.length === values.length ? times.at(-1) - times[0] : 0;
  const point = (v, i) => [padding + (duration > 0 ? (times[i] - times[0]) / duration : i / (values.length - 1)) * (w - padding * 2), h - padding - (v - lower) / height * (h - padding * 2)];
  if (bars) {
    ctx.fillStyle = color;
    values.forEach((v, i) => { if (v === null) return; const [x, y] = point(v, i); ctx.fillRect(x, y, Math.max(1, (w - 2 * padding) / values.length - 1), h - padding - y); }); return;
  }
  let connected = false;
  ctx.beginPath(); values.forEach((v, i) => { if (v === null) { connected = false; return; } const [x, y] = point(v, i); connected ? ctx.lineTo(x, y) : ctx.moveTo(x, y); connected = true; });
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
  if (markers) {
    ctx.fillStyle = color;
    values.forEach((v, i) => { if (v === null) return; const [x, y] = point(v, i); ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 2 * Math.PI); ctx.fill(); });
  }
}
