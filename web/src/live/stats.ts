import type { LiveMetrics, PlaybackMetrics } from './protocol';

export class Stats {
  private sum = 0;
  private frames = 0;
  private maximum = 0;
  private misses = 0;
  private playback: PlaybackMetrics = { underrunSamples: 0 };

  updatePlayback(metrics: PlaybackMetrics): void { this.playback = metrics; }

  add(frameMs?: number): void {
    if (frameMs === undefined || !Number.isFinite(frameMs) || frameMs < 0) return;
    this.frames++;
    this.sum += frameMs;
    this.maximum = Math.max(this.maximum, frameMs);
    if (frameMs > 20) this.misses++;
  }

  snapshot(inputSamples: number, pendingChunks: number): LiveMetrics {
    return { elapsedSeconds: inputSamples / 16000, frames: this.frames,
      meanFrameMs: this.frames ? this.sum / this.frames : 0,
      maxFrameMs: this.maximum, deadlineMisses: this.misses, pendingChunks,
      playbackUnderrunMs: this.playback.underrunSamples / 16, startupMs: this.playback.startupMs };
  }
}
