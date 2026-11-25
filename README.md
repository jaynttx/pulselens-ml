# PulseLens — Python ML project

Contactless pulse research for **jaynttx**, with a lavender, white and cream
website. Face tracking, POS, FFT analysis, pretrained FacePhys inference,
breathing, pulse-rate variability, and temporal video amplification run in
**Python**. HTML/CSS and small JavaScript modules handle the webcam and charts.

## Run on your Mac

Use Python 3.11–3.13. Stop the old `npm start` server with Ctrl+C first.

```bash
cd "/Users/jayantkaushik/Documents/Codex/2026-11-24/new-chat/work/pulselens"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://localhost:4173** in Chrome. Select a method, check consent, allow
the webcam, and capture for 30–60 seconds. Keep the terminal running.
If port 4173 is already occupied, use `python app.py --port 4174` and open
http://localhost:4174 instead. Node.js is not required to run the website.

The first start downloads the selected detector/model assets. To download all
assets before opening the camera:

```bash
python scripts/download_models.py
```

**Run system check** executes both Python face trackers and 480 recurrent model
frames, projection, SQI and PSD using artificial diagnostic input. It does not
open the webcam or present diagnostic input as a human measurement.

If Python on macOS reports a certificate error, install the certificates using
the `Install Certificates.command` supplied by your python.org installation;
do not disable HTTPS verification. The model downloader uses certifi roots.

## What moved from JavaScript to Python?

| Python file | Responsibility |
| --- | --- |
| `app.py` | Application entry point |
| `pulselens/web.py` | Local HTTP server, frame API, session lifecycle |
| `pulselens/face_models.py` | CPU adapters for pretrained BlazeFace and 478-point face landmarks |
| `pulselens/vision.py` | Face regions, forehead/cheek skin regions, adaptive pixel weights, amplification |
| `pulselens/tracking.py` | Kalman smoothing and frame timing |
| `pulselens/classical.py` | Reference-profile POS, rolling mean, FFT bandpass, Hamming spectrum and autocorrelation fusion |
| `pulselens/signals.py` | Shared resampling and long-window research-signal helpers |
| `pulselens/facephys.py` | Python LiteRT tensor adapter, recurrent state, projection, SQI/PSD |
| `pulselens/vitals.py` | Respiratory proxy and pulse-rate variability |
| `pulselens/session.py` | Connects the processing stages and independent research windows |
| `pulselens/assets.py` | Pinned pretrained model download |
| `benchmark/analyze.py` | Paired-reference evaluation |

Face models execute through Google LiteRT's Python CPU runtime. This avoids
the native MediaPipe Metal graph that can abort on a headless Mac, while
retaining MediaPipe's pretrained detector and landmark weights. The Python
adapter implements model preprocessing, anchor decoding, NMS and rotated crops.

The old JavaScript processing workers were removed. `src/app.js` captures and
sends lossless RGBA camera frames to the Python server and renders its results;
it does not extract pulse, execute neural models, or calculate research vitals.
The remaining JavaScript also draws charts and formats CSV/dashboard values.
A web browser needs this small client-side camera/interface layer.

## Research approach

- Classical: MediaPipe face landmarks → skin-filtered 32×32 forehead/cheek regions →
  256-sample colour buffer → overlapping 48-frame POS → rolling-mean removal →
  FFT bandpass → Hamming-window FFT/autocorrelation fusion. These settings match
  the pinned reference; 150 captured samples are required before estimating.
- Neural: BlazeFace → Kalman-smoothed expanded 36×36 RGB crop → pretrained
  FacePhys recurrent inference → 450 raw pulse samples → learned SQI/PSD, with
  correction with the latest smoothed frame interval, matching the reference. Learned SQI above 0.38 gates neural readings.
- Breathing uses a separate 30-second colour window. Pulse-rate variability
  needs about 60 seconds and consistent beat intervals. Panels show progress;
  stopping preserves previous results with clear historical labels.

The same research methods and pretrained assets are retained. Rewriting an
implementation does not guarantee identical numerical outputs or performance.

## Privacy and deployment

Run the server on this computer only; it binds to **127.0.0.1**. The browser sends
raw frames solely to that local Python process. Frames and measurements stay
in memory and are not logged or written to disk. Closing/stopping a session
releases its tracker and buffers. Exporting CSV is an explicit user action.

This differs from the earlier browser-only implementation: the Python server
must remain running, including during offline use. Cached model weights avoid
repeat downloads. Static-only hosting such as GitHub Pages cannot run this
Python backend. Do not expose this development server over the internet.

## Tests

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p 'test_*.py'
```

Optional frontend checks, if Node.js is installed:

```bash
npm test
npm run build
```

Tests exercise known synthetic pulse/respiration frequencies, noise rejection,
PRV, tensor/state contracts, frame decoding, Python session processing, and a
real loopback HTTP API. Camera/vision/model doubles in unit tests are labelled
as synthetic; they do not validate human accuracy. The actual pretrained
runtime is checked separately by **Run system check** or
`python scripts/check_models.py`, and by the pretrained-runtime CI job. Compare live readings
with a simultaneous reference device before reporting accuracy.

## Models and attribution

FacePhys weights/initial states are third-party pretrained assets, not trained
by this project. PulseLens by [xan-antx](https://github.com/xan-antx/pulseLens)
inspired the method comparison. Assets are pinned to revision
`e7bdd94cbe686611eb37f817bbdf48b8d770b057`.
See `THIRD_PARTY_NOTICES.md` for the model's MIT licence with Privacy Protection
Addendum. The new application code is MIT licensed. This is a student research
prototype, not a medical device or a basis for diagnoses.

## Reference comparison

See `docs/VERIFICATION.md` and `docs/verification-results.json`. The classical
Python estimator was compared against the actual pinned TypeScript estimator
on 21 controlled RGB traces. A separate tensor-by-name runner checked 480
neural frames, feature projections, SQI and PSD against the Python adapter.
These compare processing semantics, not live-human measurement accuracy.

To reproduce with a separate checkout of the reference repository:

```bash
python scripts/compare_reference.py --upstream /path/to/reference-checkout
```

This optional comparison needs Node.js 24+ to execute the reference TypeScript.
The website itself still runs with Python alone.
