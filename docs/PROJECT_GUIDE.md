# PulseLens project and viva guide

## Problem and reason for selection

Contact sensors require dedicated hardware and physical contact. A webcam can
observe small changes in reflected facial light associated with blood-volume
changes. Our research question is whether a browser-local learned pulse model
and a transparent classical method can give useful experimental estimates,
and how their errors change with lighting and movement.

## Input and output

Input: timestamped webcam RGB frames, one visible consenting participant,
and optional independently recorded reference BPM. Classical mode averages
three forehead/cheek patches. Deep mode uses a 36×36 RGB face crop, float32
values scaled to [0,1], and the elapsed time between neural input frames.
Output: a pulse waveform, spectral peak, gated BPM, heuristic quality, and
paired measurement error. No labels such as healthy/unhealthy are inferred.

## Classical algorithm

Uniformly resample timestamped RGB traces to 30 Hz. For overlapping 1.6-second
windows normalize each channel by its local mean. Form S1 = G−B and
S2 = G+B−2R. Combine h = S1 + std(S1)/std(S2) × S2. Remove each window's mean
and overlap-average. Detrend the aggregate signal. Apply a Hann taper and
evaluate spectral power between 40/60 and 200/60 Hz. Peak frequency × 60
gives BPM. Autocorrelation close to that peak provides a periodicity check.

The 0.5 BPM spectral grid is interpolation density, **not** true frequency
resolution. A 20-second window has approximately 0.05 Hz = 3 BPM resolution.
We do not claim that scanning a fine grid creates extra physiological detail.

## Deep algorithm

The third-party FacePhys state-space neural model consumes one face crop,
elapsed time, and recurrent hidden tensors. The worker carries returned states
forward with the pinned input/output mapping. It returns a scalar pulse.
The same timestamp-aware spectral estimator then estimates its repetition
rate. Workers prevent numerical processing from blocking interface input.

This is **pretrained model integration**, not training from scratch. Do not say
we collected a dataset, trained FacePhys, obtained its authors' accuracy, or
reproduced their experiments. The inference runtime and model files require
real-browser verification on a webcam-equipped device before a demo.

## Dataset and train/test discussion

This repository contains no human video training data. Established research
datasets such as UBFC-rPPG and PURE may be useful for future evaluation, subject
to their access/consent/license terms. Obtain official data rather than invent
samples. Use subject-disjoint training, validation, and test sets for any
future fine-tuning; adjacent windows from the same person must not leak
between sets. Explain the difference between evaluating a pretrained model
and claiming to train one.

## Experimental protocol

Choose multiple consenting participants and report their count. Repeat trials
under good light, dim light, speaking/movement, and post-exercise conditions.
Record device, browser, camera, light conditions, usable duration, and rejected
windows. Each method needs its own synchronized reference, even when trials
are sequential. Use equal durations, warm-up rules, and quality thresholds.
Record the reference instrument's limitations.

Report MAE = mean(|prediction−reference|), RMSE = sqrt(mean(error²)), and bias
= mean(prediction−reference). Report rejection rate and sample counts alongside
errors so hiding difficult readings does not make a method appear better.
For subject-level inference, aggregate per subject and report uncertainty;
overlapping windows are not independent participants.

## Quality and ethics

We require face presence, limited movement, workable brightness, a concentrated
spectral peak, and positive autocorrelation near its period. These checks can
still accept periodic motion or lighting artifacts. Do not interpret the
quality score as a calibrated accuracy probability. Evaluate across skin
appearances and conditions, disclose selection bias, and avoid clinical claims.
FacePhys's privacy addendum requires local inference and explicit consent.

## Viva questions

**What is rPPG?** Estimating a pulse from temporal changes in reflected light
from skin, without a contact sensor.

**Why use green?** Blood-volume changes influence visible light absorption;
green often carries useful pulse variation. POS combines all three channels
rather than relying on green alone.

**Is POS machine learning?** No. It is an analytically defined projection and
signal-processing method. FacePhys and face detection are learned models.

**Why a state-space model?** A recurrent hidden state summarizes prior frames,
giving fixed-size streaming memory. This is useful for real-time inference.

**Why timestamp resampling?** Cameras do not deliver perfectly uniform frames.
Ignoring timing errors changes the conversion between cycles and seconds.

**Why wait 20 seconds?** A longer signal window improves frequency resolution
and makes periodicity easier to distinguish from noise.

**What did you implement?** The new frontend, camera lifecycle, POS/spectral
pipeline, model adapter, quality checks, CSV export, benchmark lab, tests,
and project documentation. Pretrained weights are explicitly credited.

**Did you test medical accuracy?** No. The bundled tests check numerical logic
with synthetic signals. Human paired trials must be collected separately.

**Why no HRV?** Reliable beat-to-beat timing needs stronger validation than
estimating one dominant spectral peak, especially from ordinary webcams.
