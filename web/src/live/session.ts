import { AudioGraph } from './audio-graph';
import { BoundedQueue } from './queue';
import { Recordings } from './recording';
import { Stats } from './stats';
import { WorkerClient } from './worker-client';
import type { LiveOptions, LiveRecordings, WorkerReply } from './protocol';

/** One local research session. The processed track can feed a WebRTC sender. */
export class LiveSession {
  private readonly client = new WorkerClient();
  private readonly queue = new BoundedQueue(3);
  private readonly recordings = new Recordings();
  private readonly stats = new Stats();
  private graph?: AudioGraph;
  private accepting = true;
  private failed = false;
  private stopping?: Promise<LiveRecordings>;
  private constructor(private readonly options: LiveOptions) {}

  static async start(options: LiveOptions): Promise<LiveSession> {
    const maximum = options.maxSeconds ?? 60;
    if (!(maximum > 0 && maximum <= 60)) throw new Error('Research recordings are limited to 60 seconds.');
    if (!/^[a-f0-9]{64}$/i.test(options.manifestSha256)) throw new Error('Set the trusted model manifest SHA-256 first.');
    const session = new LiveSession(options);
    try {
      await session.client.request('init', { manifestUrl: new URL(options.manifestUrl, location.href).href,
        manifestSha256: options.manifestSha256.toLowerCase() });
      session.graph = await AudioGraph.open((audio) => session.push(audio), (error) => session.fail(error),
        (metrics) => { session.stats.updatePlayback(metrics); session.report(); });
      return session;
    } catch (error) { session.client.dispose(); throw error; }
  }

  get stream(): MediaStream {
    if (!this.graph) throw new Error('Audio capture has not started.');
    return this.graph.stream;
  }

  private push(samples: Float32Array): void {
    if (!this.accepting || this.failed) return;
    const limit = Math.floor((this.options.maxSeconds ?? 60) * 16000);
    const chunk = samples.slice(0, Math.max(0, limit - this.recordings.inputSamples));
    if (chunk.length) {
      this.recordings.add('input', chunk);
      void this.queue.push(async () => this.consume(await this.client.request('push', { samples: chunk })))
        .catch((error: Error) => this.fail(error));
    }
    if (this.recordings.inputSamples >= limit) void this.stop();
  }

  private consume(reply: WorkerReply): void {
    const samples = reply.samples ?? new Float32Array();
    this.recordings.add('restored', samples);
    this.graph?.write(samples);
    for (const duration of reply.metrics?.frameTimes ?? [reply.metrics?.frameMs]) this.stats.add(duration);
    this.report();
  }

  private report(): void {
    this.options.onMetrics?.(this.stats.snapshot(this.recordings.inputSamples, this.queue.pending));
  }

  private fail(error: Error): void {
    if (this.failed) return;
    this.failed = true;
    this.options.onError?.(error);
    void this.stop();
  }

  stop(): Promise<LiveRecordings> {
    this.stopping ??= this.complete();
    return this.stopping;
  }

  private async complete(): Promise<LiveRecordings> {
    try {
      await this.graph?.stopCapture();
      this.accepting = false;
      await this.queue.finish();
      if (!this.failed && this.recordings.inputSamples >= 640) this.consume(await this.client.request('finish'));
    } catch (error) { this.failed = true; this.options.onError?.(error as Error); }
    finally {
      this.client.dispose();
      await this.graph?.close().catch((error: Error) => { this.failed = true; this.options.onError?.(error); });
    }
    const result = this.recordings.finish();
    this.options.onStopped?.(result);
    return result;
  }
}
