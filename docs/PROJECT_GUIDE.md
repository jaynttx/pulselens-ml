# PulseLens Python project guide

## Explain the architecture

The browser captures lossless RGBA pixels and sends one frame at a time to a
Python server on the same computer. Python tracks the face, extracts pulse,
runs pretrained inference, and computes research estimates. The response
contains a bounding box, pulse sample, model features and periodic analysis.
The browser only renders those results. A single in-flight frame prevents
unbounded inference queues.

## Classical baseline

`vision.py` uses the pretrained MediaPipe pretrained face-landmark model to identify forehead and
cheeks. A skin-colour mask excludes background pixels. Adaptive spatial
weights favour locations that follow the recent pulse frequency.
`signals.py` resamples timestamps, performs overlapping 1.6-second POS, detrends,
filters at 0.67–3.33 Hz, applies a Hann taper and computes an FFT. A spectral
concentration and autocorrelation check rejects noisy windows. Zero padding
supports interpolation but does not add physical frequency resolution.

`face_models.py` uses Python LiteRT CPU inference for the same pretrained
BlazeFace and landmark weights. It letterboxes input, decodes 896 anchors,
merges overlapping detections and maps rotated face landmarks back to frame
coordinates. It avoids native Metal graph initialization on headless Macs.

## Deep learning

`facephys.py` loads pretrained FacePhys, projection, SQI and PSD models through
Google LiteRT's Python Interpreter API. It maintains 46 recurrent state tensors,
passes frame time and a normalized 36×36 RGB crop, and carries model output
states into the next frame. Tensor shapes are checked before execution; an
incompatible runtime/model fails explicitly. Kalman-smoothed face boxes and
smoothed timing reduce detector and frame jitter.

The learned quality and spectral models process 450 raw neural outputs. SQI
above 0.38 accepts a plausible neural rate; the short classical BPM gate does
not veto neural output. Correct the PSD model's 30 Hz rate for actual timing.
Model weights are external pretrained work, not a training contribution here.

## Research insights

Breathing is an experimental slow colour-variation proxy after 30 seconds.
Pulse-rate variability uses consistent detected pulse intervals from about
60 seconds. RMSSD measures successive-interval differences; SDNN measures
interval spread. PRV is not interchangeable with ECG-derived HRV. Each panel
has its own readiness checks; valid breathing is independent of the displayed
short BPM window. Historical values are labelled when capture stops or quality
falls. Temporal amplification magnifies real pixel variations and artifacts.

## Evaluation and limitations

Use simultaneous reference measurements. Record separate paired trials for
POS and FacePhys under good light, dim light, motion and other conditions.
Report sample count, MAE, RMSE and bias by condition using `benchmark/analyze.py`.
Do not treat synthetic tests or runtime checks as human validation. Lighting,
motion, camera timing, skin appearance, JPEG compression and model-domain
mismatch affect readings; the application therefore sends raw RGBA rather than
JPEG frames to the local server.

## Viva questions

- What is ML here? Pretrained FacePhys and MediaPipe; POS is analytical processing.
- What was implemented? Python processing, model adapter, local server, sessions,
  research estimates, website integration, evaluation tools and tests.
- Did you train FacePhys? No. Its external pretrained weights are attributed.
- Why keep JavaScript? Browser webcam access and chart rendering require a web
  client; the numerical and ML logic lives in Python.
- Where is data sent? Only to the user's loopback Python process, not a remote API.
