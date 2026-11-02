import { createProcessor, type AudioProcessor } from './processor';
interface Request { id: number; type: string; manifestUrl?: string; manifestSha256?: string; samples?: Float32Array }
let processor: AudioProcessor | undefined;
let pending = Promise.resolve();
async function dispatch(request: Request) {
  if (request.type === 'init') {
    if (processor) throw new Error('Worker already initialized.');
    processor = await createProcessor(request.manifestUrl!, request.manifestSha256!);
    return { type: 'ready' };
  }
  if (!processor) throw new Error('Initialize the model first.');
  if (request.type === 'push') return { type: 'audio', ...await processor.push(request.samples!) };
  if (request.type === 'finish') return { type: 'finished', ...await processor.finish() };
  if (request.type === 'reset') { processor.reset(); return { type: 'ready' }; }
  if (request.type === 'dispose') { await processor.dispose(); processor = undefined; return { type: 'finished' }; }
  throw new Error('Unknown worker request.');
}
async function respond(request: Request) {
  try {
    const response = { id: request.id, ...await dispatch(request) };
    postMessage(response, { transfer: 'samples' in response ? [response.samples.buffer] : [] });
  } catch (error) { postMessage({ id: request.id, type: 'error', message: String(error) }); }
}
globalThis.onmessage = (event: MessageEvent<Request>) => { pending = pending.then(() => respond(event.data)); };
