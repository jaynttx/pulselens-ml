# Sources and third-party notices

Reference inspiration and model distribution:
https://github.com/xan-antx/pulseLens
Pinned revision: e7bdd94cbe686611eb37f817bbdf48b8d770b057

FacePhys model assets and tensor interface are third-party work. The adapter
uses the distributed 48-input model contract. This project does not claim
ownership of, or training contributions to, the pretrained weights.
The supplied model license is reproduced below without modification.

MediaPipe Tasks Vision is distributed by Google under Apache-2.0:
https://github.com/google-ai-edge/mediapipe
LiteRT.js is distributed by Google:
https://github.com/google-ai-edge/LiteRT.js
Runtime package versions are pinned in the source. Third-party packages and
assets retain their respective licenses. Google Fonts serves DM Sans and
Manrope under the SIL Open Font License.

## FacePhys license (from apps/pulselens-deep/LICENSE)

The MIT License (MIT)

Copyright (c) 2025 Kegang Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Privacy Protection Addendum

By using or distributing this software, you agree to the following additional
terms:

1. Local Processing (No Exceptions): All biometric inference and signal
   processing must be performed exclusively on the local device. This
   requirement is absolute and cannot be waived by consent. You must NOT
   perform inference on, or offload raw user video feeds to, any external
   server or cloud service.

2. Transmission of Results: You must NOT transmit, upload, or stream derived
   physiological metrics to any external server or cloud service, UNLESS the
   data subject has given explicit, informed consent that clearly describes
   what data is collected, where it is sent, how it will be used, and how long
   it will be retained. Such consent must be freely given, specific, and
   revocable at any time.

3. Consent Requirement: You shall not use this Software to collect information
   or physiological data from any individual without their explicit consent.

4. No Backdoors in Distribution: If you redistribute or modify this code, you
   must maintain these strict local-processing guarantees. You are prohibited
   from modifying the software to surreptitiously collect or transmit user data.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
