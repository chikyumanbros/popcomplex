/**
 * Rolling time-series: org count, Simpson diversity, and structure metrics (lineages, dominance share, perimeter).
 * Drawing is decimated; left axis = population scale, right axis = 0–1 fractions.
 */

const MAX_SAMPLES = 720;
const MAX_DRAW_VERTICES = 320;

const COL_ORGS = '#5eb8c4';
const COL_LINEAGES = '#8ab4e8';
const COL_SIMPSON = '#6eb88a';
const COL_TOP_SHARE = '#e8b06a';
const COL_PERIM = '#c090d8';
const COL_GRID = 'rgba(255,255,255,0.06)';
const COL_TEXT = '#6b7a8c';
const COL_BG = '#0e1218';

export class EcologyTrendChart {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly dpr: number;
  private ticks: number[] = [];
  private orgs: number[] = [];
  private simpson: number[] = [];
  private lineages: number[] = [];
  private topShare: number[] = [];
  private perimeter: number[] = [];
  private drawPending = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('EcologyTrendChart: 2d context unavailable');
    this.ctx = ctx;
    this.dpr = 1;
    this.resize();
    if (typeof window !== 'undefined') {
      window.addEventListener('resize', () => this.resize());
    }
  }

  private resize(): void {
    const parent = this.canvas.parentElement;
    let w = parent ? parent.clientWidth : 0;
    if (w < 48) w = 240;
    const cssH = 156;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${cssH}px`;
    this.canvas.width = Math.floor(w * this.dpr);
    this.canvas.height = Math.floor(cssH * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.scheduleDraw();
  }

  scheduleDraw(): void {
    if (this.drawPending) return;
    this.drawPending = true;
    requestAnimationFrame(() => {
      this.drawPending = false;
      this.draw();
    });
  }

  /**
   * Call after a simulation step batch; `tick` is current global tick.
   * `topShare` = largest lineage / occupied cells; `perimeter` = boundary faces / (4 * occupied).
   */
  sample(
    tick: number,
    orgCount: number,
    simpsonDiversity: number,
    uniqueLineages: number,
    topLineageShare: number,
    perimeterRatio: number,
  ): void {
    this.ticks.push(tick);
    this.orgs.push(orgCount);
    this.simpson.push(simpsonDiversity);
    this.lineages.push(uniqueLineages);
    this.topShare.push(topLineageShare);
    this.perimeter.push(perimeterRatio);
    while (this.ticks.length > MAX_SAMPLES) {
      this.ticks.shift();
      this.orgs.shift();
      this.simpson.shift();
      this.lineages.shift();
      this.topShare.shift();
      this.perimeter.shift();
    }
    this.scheduleDraw();
  }

  private strokeSeries(
    m: number,
    idxAt: (j: number) => number,
    xAt: (j: number) => number,
    yAt: (i: number) => number,
    strokeStyle: string,
    lineWidth: number,
  ) {
    const ctx = this.ctx;
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    for (let j = 0; j < m; j++) {
      const i = idxAt(j);
      const x = xAt(j);
      const y = yAt(i);
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  draw(): void {
    const ctx = this.ctx;
    const w = this.canvas.width / this.dpr;
    const h = this.canvas.height / this.dpr;
    ctx.fillStyle = COL_BG;
    ctx.fillRect(0, 0, w, h);

    const padL = 38;
    const padR = 38;
    const padT = 22;
    const padB = 28;
    const plotW = Math.max(1, w - padL - padR);
    const plotH = Math.max(1, h - padT - padB);

    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = COL_TEXT;
    ctx.textAlign = 'left';
    ctx.fillText('Ecology & structure (recent)', 6, 14);

    const n = this.ticks.length;
    if (n < 2) {
      ctx.fillStyle = COL_TEXT;
      ctx.fillText('Run simulation to plot…', padL, padT + plotH * 0.45);
      return;
    }

    let minO = Infinity;
    let maxO = -Infinity;
    let minL = Infinity;
    let maxL = -Infinity;
    let minR = Infinity;
    let maxR = -Infinity;
    for (let i = 0; i < n; i++) {
      const o = this.orgs[i]!;
      const l = this.lineages[i]!;
      const s = this.simpson[i]!;
      const t = this.topShare[i]!;
      const p = this.perimeter[i]!;
      if (o < minO) minO = o;
      if (o > maxO) maxO = o;
      if (l < minL) minL = l;
      if (l > maxL) maxL = l;
      if (s < minR) minR = s;
      if (s > maxR) maxR = s;
      if (t < minR) minR = t;
      if (t > maxR) maxR = t;
      if (p < minR) minR = p;
      if (p > maxR) maxR = p;
    }
    minO = Math.min(0, minO);
    maxO = Math.max(10, maxO * 1.08);
    minL = Math.min(0, minL);
    maxL = Math.max(2, maxL * 1.12);
    const leftMax = Math.max(maxO, maxL);
    const leftMin = 0.0;

    minR = Math.max(0, minR - 0.03);
    maxR = Math.min(1, Math.max(maxR + 0.03, minR + 0.06));

    const spanLeft = leftMax - leftMin || 1;
    const spanR = maxR - minR || 1;

    ctx.strokeStyle = COL_GRID;
    ctx.lineWidth = 1;
    for (let g = 0; g <= 3; g++) {
      const y = padT + (plotH * g) / 3;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
    }

    const m = Math.min(n, MAX_DRAW_VERTICES);
    const idxAt = (j: number) => Math.round((j / Math.max(1, m - 1)) * (n - 1));
    const xAt = (j: number) => {
      const i = idxAt(j);
      return padL + (i / (n - 1)) * plotW;
    };
    const yLeft = (v: number) => padT + plotH - ((v - leftMin) / spanLeft) * plotH;
    const yRight = (v: number) => padT + plotH - ((v - minR) / spanR) * plotH;

    this.strokeSeries(m, idxAt, xAt, (i) => yLeft(this.orgs[i]!), COL_ORGS, 1.5);
    this.strokeSeries(m, idxAt, xAt, (i) => yLeft(this.lineages[i]!), COL_LINEAGES, 1.25);
    this.strokeSeries(m, idxAt, xAt, (i) => yRight(this.simpson[i]!), COL_SIMPSON, 1.5);
    this.strokeSeries(m, idxAt, xAt, (i) => yRight(this.topShare[i]!), COL_TOP_SHARE, 1.15);
    this.strokeSeries(m, idxAt, xAt, (i) => yRight(this.perimeter[i]!), COL_PERIM, 1.15);

    ctx.fillStyle = COL_TEXT;
    ctx.textAlign = 'right';
    ctx.fillText(String(Math.round(leftMax)), padL - 4, padT + 9);
    ctx.fillText(String(Math.round(leftMin)), padL - 4, padT + plotH);
    ctx.textAlign = 'left';
    ctx.fillText(maxR.toFixed(2), padL + plotW + 4, padT + 9);
    ctx.fillText(minR.toFixed(2), padL + plotW + 4, padT + plotH);

    const legY = padT + plotH + 14;
    const dot = (x: number, col: string, label: string) => {
      ctx.fillStyle = col;
      ctx.fillRect(x, legY - 6, 7, 3);
      ctx.fillStyle = COL_TEXT;
      ctx.fillText(label, x + 10, legY - 2);
    };
    ctx.textAlign = 'left';
    ctx.font = '9px ui-monospace, monospace';
    let lx = padL;
    dot(lx, COL_ORGS, 'orgs');
    lx += 48;
    dot(lx, COL_LINEAGES, 'lin');
    lx += 38;
    dot(lx, COL_SIMPSON, 'Simp');
    lx += 42;
    dot(lx, COL_TOP_SHARE, 'top');
    lx += 34;
    dot(lx, COL_PERIM, 'perim');
  }
}
