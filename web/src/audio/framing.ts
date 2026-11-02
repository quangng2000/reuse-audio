/** Centered reflection matching the native stream; bounded to two frames. */
export class StreamFrames {
  private buffer: number[] = [];
  private recent: number[] = [];
  private initialized = false;
  private finished = false;
  private frames = 0;
  received = 0;
  push(samples: Float32Array): Float32Array[] {
    if (this.finished) throw new Error('Stream finished.');
    if (samples.length > 320 || !samples.every(Number.isFinite)) throw new Error('Expected at most 320 finite samples.');
    this.received += samples.length;
    this.recent = [...this.recent, ...samples].slice(-641);
    this.buffer.push(...samples);
    if (!this.initialized && this.buffer.length >= 640) {
      this.buffer.unshift(...this.buffer.slice(1, 321).reverse()); this.initialized = true;
    }
    return this.initialized ? this.drain(Infinity) : [];
  }
  finish(): Float32Array[] {
    if (this.finished || !this.initialized) throw new Error('Finish requires at least 640 samples and an open stream.');
    this.finished = true;
    this.buffer.push(...this.recent.slice(0, -1).reverse());
    return this.drain(1 + Math.ceil(this.received / 320) - this.frames);
  }
  private drain(limit: number): Float32Array[] {
    const frames: Float32Array[] = [];
    while (this.buffer.length >= 640 && frames.length < limit) {
      frames.push(new Float32Array(this.buffer.slice(0, 640)));
      this.buffer.splice(0, 320); this.frames++;
    }
    return frames;
  }
}
