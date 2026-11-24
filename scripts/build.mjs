import { mkdir, cp, rm } from 'node:fs/promises';
await rm('dist', { recursive: true, force: true }); await mkdir('dist');
for (const file of ['index.html', 'favicon.svg', 'manifest.webmanifest', 'sw.js', 'src', 'models', 'THIRD_PARTY_NOTICES.md']) {
  await cp(file, `dist/${file}`, { recursive: true });
}
await cp('dist/index.html', 'dist/404.html');
console.log('Static build created in dist/. Serve over HTTPS or localhost.');
