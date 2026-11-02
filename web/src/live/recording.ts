import type { LiveRecordings } from './protocol';

function encodeWave(chunks: Float32Array[], samples: number, gain: number): Blob {
  const bytes = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(bytes);
  const text = (offset: number, value: string) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, bytes.byteLength - 8, true); text(8, 'WAVEfmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 16000, true); view.setUint32(28, 32000, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, samples * 2, true);
  let offset = 44;
  for (const chunk of chunks) for (const sample of chunk) {
    const safe = Math.max(-1, Math.min(1, sample * gain));
    view.setInt16(offset, Math.round(safe * (safe < 0 ? 32768 : 32767)), true); offset += 2;
  }
  return new Blob([bytes], { type: 'audio/wav' });
}

export class Recordings {
  private input: Float32Array[] = [];
  private restored: Float32Array[] = [];
  private peak = 1;
  inputSamples = 0;
  restoredSamples = 0;

  add(kind: 'input' | 'restored', samples: Float32Array): void {
    if (!samples.every(Number.isFinite)) throw new Error('Audio contains nonfinite samples.');
    for (const sample of samples) this.peak = Math.max(this.peak, Math.abs(sample));
    this[kind].push(samples.slice());
    if (kind === 'input') this.inputSamples += samples.length;
    else this.restoredSamples += samples.length;
  }

  finish(): LiveRecordings {
    const playbackGain = Math.min(1, 0.99 / this.peak);
    return { input: encodeWave(this.input, this.inputSamples, playbackGain),
      restored: encodeWave(this.restored, this.restoredSamples, playbackGain),
      inputSamples: this.inputSamples, restoredSamples: this.restoredSamples, playbackGain };
  }
}
