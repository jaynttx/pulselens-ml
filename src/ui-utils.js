// Display-only helpers: no pulse extraction or ML runs in JavaScript.
export const mean = a => a.reduce((s, v) => s + v, 0) / (a.length || 1);
export const std = a => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

export function metrics(rows) {
  if (!rows.length) return null;
  const e = rows.map(r => r.app - r.reference);
  return { n: e.length, mae: mean(e.map(Math.abs)), rmse: Math.sqrt(mean(e.map(v => v * v))), bias: mean(e) };
}

export function csv(rows) {
  // Spreadsheet formula escaping applies only to strings, preserving numeric negatives.
  return rows.map(row => row.map(value => {
    let s = String(value ?? '');
    if (typeof value === 'string' && /^[=+\-@\t\r]/.test(s)) s = "'" + s;
    return '"' + s.replaceAll('"', '""') + '"';
  }).join(',')).join('\r\n');
}
