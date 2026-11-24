# PulseLens · Contactless pulse research

A fresh student-project implementation of webcam heart-rate estimation with a
lavender, white, and cream frontend. Prepared for **jaynttx**.

## Run it

Requirements: Node.js 20+ and a webcam. Python 3.9+ is optional for model caching
and benchmark analysis. No npm packages need installing.

```bash
npm start
```

Open **http://localhost:4173** in Chrome or Edge. Check the local-processing
consent box, select Classical POS or Deep learning, and start the camera.
Face a light source, sit still, and allow 20–30 seconds. Stopping, changing tabs,
revoking consent, or navigating away releases the camera.

The first start needs internet access for MediaPipe. Deep mode also downloads
FacePhys model assets and the LiteRT WASM runtime. To cache the weights first:

```bash
npm run models
```

If asset downloads fail, the app reports an error. It never substitutes fake
BPM values or silently changes the selected method. This release does **not**
guarantee offline operation. Camera access requires localhost or HTTPS.

## Features

- Original responsive frontend with lavender, white, and cream styling.
- MediaPipe face detection and forehead/cheek regions for colour sampling.
- Original overlapping POS implementation with timestamp-aware resampling.
- Browser-local pretrained FacePhys SSM inference in a dedicated worker.
- Rolling pulse waveform, frequency spectrum, and accepted-reading trend.
- Motion, lighting, periodicity, and signal-quality checks; unreliable values hidden.
- Start/stop, consent withdrawal, cancellation, and camera lifecycle handling.
- CSV export of session samples and accepted BPM estimates.
- Paired benchmark lab with MAE, RMSE, bias, CSV export, and Python analysis.
- Deterministic signal-processing tests and a GitHub Actions check workflow.

## What is ML here?

Deep mode uses the **pretrained FacePhys state-space model**. Model weights and
initial states are third-party assets, not trained by Jayant or this project.
The neural worker implements its published/distributed tensor interface. The
POS baseline is signal processing, not a trained ML estimator. Face detection
uses a separate pretrained MediaPipe model.

Our implementation includes the application, UI, POS/spectral routines, inference
adapter, session handling, tests, evaluation workflow, and documentation. It is
inspired by [xan-antx/pulseLens](https://github.com/xan-antx/pulseLens), rather than
a fork or a claim of authorship over its model. The model revision is pinned to
`e7bdd94cbe686611eb37f817bbdf48b8d770b057`. See THIRD_PARTY_NOTICES.md.

This is not a feature-identical clone: the reference's learned SQI, attention
projection, hidden-state visualizations, breathing/HRV, video amplification,
and offline service worker are not implemented. This version uses documented
heuristic quality checks shared across methods, allowing a simpler student
codebase and explicit uncertainty.

## Verify and evaluate

```bash
npm test
python3 -m unittest discover -s tests -p 'test_*.py'
npm run build
```

Tests recover known synthetic pulse frequencies and check noise rejection,
timestamp resampling, POS recovery, CSV escaping, and metric calculations.
Synthetic tests prove numerical behavior, **not human measurement accuracy**.

1. Run one method under good light, dim light, motion, and after exercise.
2. Use a reference device simultaneously with each app observation. Record a
   fresh reference when changing methods; do not reuse a stale watch reading.
3. Add paired app/reference observations in Benchmark lab and export CSV.
4. Analyze: `python3 benchmark/analyze.py pulselens-benchmark.csv`.
5. Report trial counts and per-condition MAE/RMSE/bias. No human benchmark data
   or claimed clinical accuracy is bundled.

## Build and publish

`npm run build` writes the static app to `dist/`. Host that folder over HTTPS.
No server-side biometric inference or database is required. To create a GitHub
repository after authenticating the GitHub CLI:

```bash
gh auth login -h github.com
git init -b main
git add .
git commit -m "Build original PulseLens student research app"
gh repo create jaynttx/pulselens-ml --private --source=. --remote=origin --push
```

If a repository with that name already exists, add its remote and push a new
branch rather than overwriting existing work. Make it public only when desired.

## Project layout

| Path | Purpose |
| --- | --- |
| `index.html`, `src/style.css` | Frontend |
| `src/app.js` | Camera, face sampling, sessions, benchmark UI |
| `src/signal.js` | POS, uniform resampling, detrending, spectra, metrics |
| `src/analysis.worker.js` | Background signal analysis |
| `src/deep.worker.js` | Pretrained FacePhys inference |
| `scripts/download_models.py` | Optional pinned asset download |
| `benchmark/analyze.py` | Paired evaluation |
| `docs/PROJECT_GUIDE.md` | Explanation, study design, viva preparation |

## Limits and privacy

Research only; not a medical device. Motion, lighting, camera frame timing,
skin appearance, and training-domain differences can cause wrong estimates.
Quality is a periodicity heuristic, not calibrated medical confidence. HRV,
oxygen saturation, diagnoses, and stress claims are intentionally absent.
Webcam frames, pulse samples, and inferred BPM remain in local tab memory.
Only explicit CSV downloads persist measurements. External CDN requests load
model/runtime assets and fonts; no analytics or physiological uploads exist.

Fresh application code is MIT licensed. FacePhys remains subject to its own
MIT License with Privacy Protection Addendum, reproduced in the notices.
