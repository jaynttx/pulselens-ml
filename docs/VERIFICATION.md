# Processing verification — 25 November 2026

Live-capture update: the website now uses `estimate_live_rgb` with timestamp
resampling, a 10–20 second window, overlap-normalized POS and shared spectral /
autocorrelation quality checks. The pinned `estimate_rgb` comparison profile
below is retained separately. New regression tests cover 48, 60, 77, 120, 150
and 180 BPM under irregular capture intervals and illumination drift; respiratory
trace serialization and frontend trend rendering in both modes are also checked.
These synthetic checks do not determine the correct BPM for a live user.

This report verifies implementation behaviour on controlled inputs. It does
**not** establish live-human accuracy or identical camera behaviour to the
original. Raw results are in `verification-results.json`.

## What was corrected

- Neural PSD correction now uses the current smoothed frame interval, as in the
  reference, rather than a 450-sample average that could lag changes in FPS.
- Neural face crops are rounded to 8-bit RGB before inference, matching browser
  canvas readback instead of retaining fractional interpolation values.
- Classical processing uses the reference's 256-sample buffer, 48-frame POS,
  60-sample rolling mean, FFT bandpass, Hamming spectrum, FFT/autocorrelation
  fusion and confidence setting. The skin sampling grid is 32×32 per region.
- A brief capture stall does not erase history; a prolonged gap or face loss
  still starts a fresh signal window. Previous accepted session results remain.
- A stopped Python server produces restart instructions instead of only
  “Failed to fetch”. Keep its terminal open while using the app.

## Direct classical comparison

`scripts/compare_reference.py` reads the separate pinned upstream checkout and
executes its TypeScript estimator via Node, without vendoring source into this
project. It compares 21 RGB traces across 7 frequencies and 3 input lengths.

All acceptance decisions matched. Maximum BPM difference was
`2.84e-14`, and maximum confidence difference was `9.99e-16` (floating-point
rounding). This establishes agreement on those traces, not sensor accuracy.

## Neural tensor-contract comparison

The Python adapter and an independently bound LiteRT runner processed the same
480 controlled RGB crops. The independent runner binds state tensors by names
rather than reusing the adapter's positional mapping. Pulse, projected features,
SQI and PSD rate differences were zero for those inputs.

This checks the distributed model contract and recurrent state handling. It is
not a comparison of Python CPU kernels with the original browser's WASM kernels
on real camera images, and does not test differences in their face detectors.

## Motion, occlusion and processing test

A public MediaPipe test portrait was shifted horizontally/vertically by a few
pixels for 1,900 frames per method. Twenty frames deliberately replaced it with
a blank image. Both methods detected all 1,880 visible-portrait frames and
rejected the 20 blank frames. Signal buffers retained approximately 60 seconds
and continued periodic analysis after the brief occlusion.

On this Mac, measured serial processing times were:

| Method | Median/frame | 95th percentile/frame |
| --- | --- | --- |
| Classical POS | 11.770 ms | 12.797 ms |
| FacePhys | 3.896 ms | 4.471 ms |

These exclude browser/network overhead and are specific to this machine. The
portrait has no real pulse video or physiological reference; inferred rates
from this fixture are deliberately not reported as measurements.

Reproduce with `python scripts/check_motion.py --image /path/to/portrait.jpg`.
JPEG fixture decoding optionally requires Pillow. The portrait source URL is
recorded in the JSON report; it is not bundled in this repository.

## Remaining live check

The automated browser reached the camera permission step, but received no video
stream before permission timed out. Live BPM, PRV, breathing, lighting response
and movement tolerance therefore remain unverified. Allow camera access, run
Deep learning for at least 60 seconds, and compare with a simultaneous reference
sensor before claiming accuracy. Agreement with the original estimator alone
also shares that estimator's limitations; periodic motion/lighting may pass its
quality checks.
