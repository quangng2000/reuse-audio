/** Serial, bounded inference: never skip recurrent-state updates silently. */
export class BoundedQueue {
  private tail: Promise<unknown> = Promise.resolve();
  private count = 0;
  private isClosed = false;
  private failure?: unknown;

  constructor(private readonly limit = 3) {
    if (!Number.isInteger(limit) || limit < 1) throw new Error('Invalid queue limit.');
  }

  get pending(): number { return this.count; }

  push<T>(run: () => Promise<T>): Promise<T> {
    if (this.isClosed) return Promise.reject(new Error('Audio queue is closed.'));
    if (this.count >= this.limit) return Promise.reject(new Error(
      'Inference is falling behind live audio. Capture stopped instead of accumulating delay.',
    ));
    this.count++;
    const result = this.tail.then(() => {
      if (this.failure) throw this.failure;
      return run();
    });
    this.tail = result.catch((error) => { this.failure = error; this.isClosed = true; })
      .finally(() => { this.count--; });
    return result;
  }

  async finish(): Promise<void> {
    this.isClosed = true;
    await this.tail;
  }
}
