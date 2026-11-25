// Each research measurement has its own readiness/quality result. A hidden BPM
// must not hide a valid respiratory estimate or erase a completed session.
export class ResearchPanels {
  constructor(find, drawTrace = () => {}) { this.find = find; this.drawTrace = drawTrace; this.reset(false); }
  reset(draw = true) {
    this.capture = { colorSeconds: 0, pulseSeconds: 0 };
    this.current = {}; this.last = {}; this.trace = []; this.paused = false; this.active = false;
    if (draw) this.render();
  }
  clearWindow() { this.capture = { colorSeconds: 0, pulseSeconds: 0 }; this.current = {}; this.trace = []; this.render(); }
  progress(capture) { this.capture = capture; this.paused = false; this.active = true; this.render(); }
  update(result) {
    this.active = true;
    this.capture = result.capture;
    this.trace = result.breathing?.waveform || [];
    this.current = { breathing: result.breathing?.valid ? result.breathing : null,
      variability: result.variability?.valid ? result.variability : null };
    for (const key of ['breathing', 'variability']) if (this.current[key]) this.last[key] = this.current[key];
    this.render();
  }
  stop() { this.paused = true; this.render(); }
  render() {
    this.drawTrace(this.trace, this.paused);
    const put = (id, text) => { if (this.find(id).textContent !== text) this.find(id).textContent = text; };
    const { colorSeconds = 0, pulseSeconds = 0 } = this.capture;
    const breathing = this.current.breathing || this.last.breathing;
    const variability = this.current.variability || this.last.variability;
    const prefix = current => this.paused ? 'Session stopped · last accepted window. ' : current ? '' : 'Last accepted window · waiting for a new usable signal. ';
    put('breathing-rate', breathing ? breathing.rate.toFixed(1) : '—');
    put('hrv-rmssd', variability ? variability.rmssd.toFixed(1) : '—');
    put('hrv-sdnn', variability ? variability.sdnn.toFixed(1) : '—');
    put('breathing-status', breathing ? prefix(this.current.breathing) + 'Experimental respiratory proxy; movement and lighting can affect it.' :
      this.paused ? 'Session stopped before a usable respiratory window was obtained.' : !this.active ? 'Start the camera to collect a 30-second facial colour window.' : colorSeconds < 30 ? `Collecting facial colour · ${Math.floor(colorSeconds)} / 30 seconds.` : '30-second window ready; no clear respiratory rhythm detected yet.');
    put('hrv-status', variability ? prefix(this.current.variability) + 'Pulse-rate variability, not ECG-derived HRV.' :
      this.paused ? 'Session stopped before usable beat intervals were obtained.' : !this.active ? 'Start the camera to collect about 60 seconds of pulse.' : pulseSeconds < 59.5 ? `Collecting pulse · ${Math.min(60, Math.round(pulseSeconds))} / 60 seconds.` : '60-second window ready; waiting for consistent beat intervals and a usable pulse rate.');
    put('beat-count', variability ? `${variability.beatCount} detected beats · ${variability.rejected} rejected intervals` : 'Beat intervals appear after a usable pulse window.');
  }
}
