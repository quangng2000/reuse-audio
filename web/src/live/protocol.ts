export interface InferenceMetrics { frameMs?: number; frameTimes?: number[] }
export interface WorkerReply {
  id: number;
  type: 'ready' | 'audio' | 'finished' | 'error';
  samples?: Float32Array;
  metrics?: InferenceMetrics;
  message?: string;
}

export interface LiveMetrics {
  elapsedSeconds: number;
  frames: number;
  meanFrameMs: number;
  maxFrameMs: number;
  deadlineMisses: number;
  pendingChunks: number;
  playbackUnderrunMs: number;
  startupMs?: number;
}

export interface PlaybackMetrics { underrunSamples: number; startupMs?: number }

export interface LiveOptions {
  manifestUrl: string;
  manifestSha256: string;
  onMetrics?: (metrics: LiveMetrics) => void;
  onError?: (error: Error) => void;
  onStopped?: (recordings: LiveRecordings) => void;
  maxSeconds?: number;
}

export interface LiveRecordings {
  input: Blob;
  restored: Blob;
  inputSamples: number;
  restoredSamples: number;
  playbackGain: number;
}
