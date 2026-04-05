export class StatsTracker {
  private fpsBuffer: number[] = [];
  private lastFrameTime = performance.now();

  recordFrame() {
    const now = performance.now();
    this.fpsBuffer.push(1000 / (now - this.lastFrameTime));
    this.lastFrameTime = now;
    if (this.fpsBuffer.length > 60) this.fpsBuffer.shift();
  }

  get fps(): number {
    if (this.fpsBuffer.length === 0) return 0;
    return this.fpsBuffer.reduce((a, b) => a + b) / this.fpsBuffer.length;
  }
}
