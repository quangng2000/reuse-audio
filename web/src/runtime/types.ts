export interface Port { name: string; shape: number[]; dtype: 'float32' }
export interface Asset { file: string; sha256: string; bytes: number }
export interface Segment extends Asset { kind: 'onnx'; inputs: Port[]; outputs: Port[] }
export interface Scan {
  kind: 'scan'; inputs: Record<'u'|'dt'|'b'|'c'|'z'|'a'|'d', string>;
  output: string; bins: number; channels: number; components: number;
}
export interface Bundle {
  schemaVersion: 1; modelId: string; layers: number; sampleRate: number;
  fftSize: number; hopSize: number; license: string; researchOnly: boolean;
  checkpointSha256: string; steps: (Segment | Scan)[];
  constants: (Asset & Port)[];
  aliases?: { name: string; source: string }[];
  states: { inputName: string; outputName: string; shape: number[] }[];
}
