# Model assets

The application uses the pretrained FacePhys state-space model distributed by
`xan-antx/pulseLens`, pinned to commit
`e7bdd94cbe686611eb37f817bbdf48b8d770b057`.

Run `npm run models` to cache `model.tflite` and `state.gz` locally. Otherwise,
the Deep worker requests those exact assets from the pinned public source.
Model files are excluded from Git to keep this original implementation small.
Keep `THIRD_PARTY_NOTICES.md` with any distribution. Weights are not ours and
were not trained in this project. No webcam or derived measurements are sent
when downloading models.

The MediaPipe face detector and LiteRT WASM runtime load from their official
Google model host and jsDelivr, respectively. This release does not promise
offline operation.
