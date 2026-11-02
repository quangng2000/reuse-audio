import type { WorkerReply } from './protocol';

type Waiter = {
  resolve: (value: WorkerReply) => void;
  reject: (error: Error) => void;
  timeout: ReturnType<typeof setTimeout>;
};

export class WorkerClient {
  private id = 0;
  private readonly waiting = new Map<number, Waiter>();
  private readonly worker = new Worker(new URL('../worker/index.ts', import.meta.url), { type: 'module' });
  private disposed = false;

  constructor() {
    this.worker.onmessage = ({ data }: MessageEvent<WorkerReply>) => this.receive(data);
    this.worker.onerror = (event) => this.dispose(new Error(event.message || 'Inference worker failed.'));
  }

  request(type: string, fields: Record<string, unknown> = {}): Promise<WorkerReply> {
    if (this.disposed) return Promise.reject(new Error('Inference worker is closed.'));
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => this.dispose(new Error('Inference worker timed out.')), type === 'init' ? 120000 : 15000);
      this.waiting.set(id, { resolve, reject, timeout });
      try { this.worker.postMessage({ id, type, ...fields }); }
      catch (error) { clearTimeout(timeout); this.waiting.delete(id); reject(error); }
    });
  }

  private receive(reply: WorkerReply): void {
    const waiter = this.waiting.get(reply.id);
    if (!waiter) return;
    clearTimeout(waiter.timeout);
    this.waiting.delete(reply.id);
    if (reply.type === 'error') waiter.reject(new Error(reply.message || 'Inference failed.'));
    else waiter.resolve(reply);
  }

  dispose(error = new Error('Inference worker closed.')): void {
    if (this.disposed) return;
    this.disposed = true;
    this.worker.terminate();
    for (const waiter of this.waiting.values()) { clearTimeout(waiter.timeout); waiter.reject(error); }
    this.waiting.clear();
  }
}
