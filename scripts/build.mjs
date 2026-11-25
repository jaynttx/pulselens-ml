import { mkdir, cp, rm } from 'node:fs/promises';
await rm('dist', { recursive: true, force: true }); await mkdir('dist');
for (const file of ['app.py', 'requirements.txt', 'pulselens', 'index.html', 'favicon.svg', 'manifest.webmanifest', 'src', 'models', 'THIRD_PARTY_NOTICES.md']) {
  await cp(file, `dist/${file}`, { recursive: true });
}
console.log('Python website packaged in dist/. Run python app.py inside that folder; static-only hosting cannot run inference.');
