class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.capture = new Float32Array(320);
    this.captureLength = 0;
    this.isCapturing = true;
    this.output = [];
    this.outputOffset = 0;
    this.processedSamples = 0;
    this.underrunSamples = 0;
    this.startupMs = undefined;
    this.port.onmessage = ({ data }) => this.receive(data);
  }

  receive(data) {
    if (data.type === 'audio') {
      if (this.output.length >= 8) this.port.postMessage({ type: 'error', message: 'Playback queue overflow.' });
      else if (data.samples.length) this.output.push(data.samples);
    }
    if (data.type === 'stop') {
      this.isCapturing = false;
      if (this.captureLength) this.sendCapture(this.capture.slice(0, this.captureLength));
      this.port.postMessage({ type: 'stopped' });
    }
  }

  sendCapture(samples) {
    this.port.postMessage({ type: 'input', samples }, [samples.buffer]);
  }

  takeInput(input) {
    if (!this.isCapturing || !input) return;
    for (let i = 0; i < input.length; i++) {
      this.capture[this.captureLength++] = input[i];
      if (this.captureLength === 320) {
        this.sendCapture(this.capture);
        this.capture = new Float32Array(320);
        this.captureLength = 0;
      }
    }
  }

  fillOutput(output) {
    output.fill(0);
    if (this.output.length && this.startupMs === undefined) this.startupMs = this.processedSamples / 16;
    let written = 0;
    while (written < output.length && this.output.length) {
      output[written++] = this.output[0][this.outputOffset++];
      if (this.outputOffset === this.output[0].length) {
        this.output.shift();
        this.outputOffset = 0;
      }
    }
    if (this.isCapturing && this.startupMs !== undefined) this.underrunSamples += output.length - written;
    this.processedSamples += output.length;
    if (this.processedSamples % 3200 === 0) this.port.postMessage({ type: 'playback',
      underrunSamples: this.underrunSamples, startupMs: this.startupMs });
  }

  process(inputs, outputs) {
    this.takeInput(inputs[0]?.[0]);
    this.fillOutput(outputs[0][0]);
    return true;
  }
}

registerProcessor('reuse-capture', CaptureProcessor);
