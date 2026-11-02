import type { PlaybackMetrics } from './protocol';
import workletUrl from './capture-worklet.js?url&no-inline';
type CaptureHandler = (samples: Float32Array) => void;
type PlaybackHandler = (metrics: PlaybackMetrics) => void;

export class AudioGraph {
  private stopAck?: () => void;
  private closed = false;

  private constructor(
    private readonly context: AudioContext,
    private readonly microphone: MediaStream,
    private readonly node: AudioWorkletNode,
    readonly stream: MediaStream,
  ) {}

  static async open(onInput: CaptureHandler, onError: (error: Error) => void,
    onPlayback: PlaybackHandler): Promise<AudioGraph> {
    const context = new AudioContext({ sampleRate: 16000, latencyHint: 'interactive' });
    let microphone: MediaStream | undefined;
    try {
      if (context.sampleRate !== 16000) throw new Error('This browser cannot create a 16 kHz audio context.');
      await context.resume();
      await context.audioWorklet.addModule(workletUrl);
      microphone = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
      } });
      return AudioGraph.connect(context, microphone, onInput, onError, onPlayback);
    } catch (error) {
      microphone?.getTracks().forEach((track) => track.stop());
      if (context.state !== 'closed') await context.close();
      throw error;
    }
  }

  private static connect(context: AudioContext, microphone: MediaStream,
    onInput: CaptureHandler, onError: (error: Error) => void, onPlayback: PlaybackHandler): AudioGraph {
    const node = new AudioWorkletNode(context, 'reuse-capture', {
      outputChannelCount: [1], channelCount: 1, channelCountMode: 'explicit', channelInterpretation: 'speakers',
    });
    const destination = context.createMediaStreamDestination();
    const graph = new AudioGraph(context, microphone, node, destination.stream);
    context.createMediaStreamSource(microphone).connect(node);
    node.connect(destination);
    node.port.onmessage = ({ data }) => {
      if (data.type === 'input') onInput(data.samples);
      if (data.type === 'stopped') graph.stopAck?.();
      if (data.type === 'error') onError(new Error(data.message));
      if (data.type === 'playback') onPlayback(data);
    };
    node.onprocessorerror = () => onError(new Error('Browser audio processing failed.'));
    return graph;
  }

  write(samples: Float32Array): void {
    if (!this.closed && samples.length) this.node.port.postMessage({ type: 'audio', samples });
  }

  async stopCapture(): Promise<void> {
    if (this.closed) return;
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Audio capture did not acknowledge stop.')), 1500);
      this.stopAck = () => { clearTimeout(timeout); resolve(); };
      this.node.port.postMessage({ type: 'stop' });
    });
    this.microphone.getTracks().forEach((track) => track.stop());
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.microphone.getTracks().forEach((track) => track.stop());
    this.stream.getTracks().forEach((track) => track.stop());
    this.node.disconnect();
    this.node.port.close();
    if (this.context.state !== 'closed') await this.context.close();
  }
}
