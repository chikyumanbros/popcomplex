/**
 * Rolling time-series for ecology / structure. Two Y scales: left = counts, right = unitless [0,1].
 * Series list, colors, and axes are defined in `ECOLOGY_SERIES` (single source of truth).
 */

/**
 * Rolling buffer length. With `CHART_SAMPLE_EVERY_SIM_TICK = 2` in `main.ts`, this spans about
 * `(MAX_SAMPLES - 1) * 2` simulation ticks on the x-axis (~100k ticks at 50_001 samples).
 */
const MAX_SAMPLES = 50_001;
const MAX_DRAW_VERTICES = 320;
/** Smallest number of samples shown when zoomed in (needs ≥2 for a polyline). */
const MIN_TIME_WINDOW_SAMPLES = 2;

const PAD_L = 38;
const PAD_R = 38;
const PAD_T = 22;
const PAD_B = 36;

const COL_GRID = 'rgba(255,255,255,0.06)';
const COL_TEXT = '#6b7a8c';
const COL_BG = '#0e1218';

const STORAGE_KEY = 'popcomplex.ecologyChart.series';

export type EcologySeriesId = 'orgs' | 'lineages' | 'simpson' | 'topShare' | 'perimeter';

type SeriesAxis = 'left' | 'right';

interface EcologySeriesDef {
  readonly id: EcologySeriesId;
  /** Checkbox / settings row */
  readonly toggleLabel: string;
  /** Canvas legend (short) */
  readonly legendShort: string;
  readonly color: string;
  readonly lineWidth: number;
  readonly axis: SeriesAxis;
}

/** Display order = draw order (bottom series drawn first where strokes overlap). */
export const ECOLOGY_SERIES: readonly EcologySeriesDef[] = [
  {
    id: 'orgs',
    toggleLabel: 'Organisms',
    legendShort: 'orgs',
    color: '#5eb8c4',
    lineWidth: 1.5,
    axis: 'left',
  },
  {
    id: 'lineages',
    toggleLabel: 'Lineages',
    legendShort: 'lineages',
    color: '#8ab4e8',
    lineWidth: 1.25,
    axis: 'left',
  },
  {
    id: 'simpson',
    toggleLabel: 'Simpson',
    legendShort: 'Simpson',
    color: '#6eb88a',
    lineWidth: 1.5,
    axis: 'right',
  },
  {
    id: 'topShare',
    toggleLabel: 'Top lineage',
    legendShort: 'top share',
    color: '#e8b06a',
    lineWidth: 1.15,
    axis: 'right',
  },
  {
    id: 'perimeter',
    toggleLabel: 'Perimeter',
    legendShort: 'perimeter',
    color: '#c090d8',
    lineWidth: 1.15,
    axis: 'right',
  },
] as const;

function defaultVisibility(): Record<EcologySeriesId, boolean> {
  const v = {} as Record<EcologySeriesId, boolean>;
  for (const s of ECOLOGY_SERIES) v[s.id] = true;
  return v;
}

function loadVisibility(): Record<EcologySeriesId, boolean> {
  const def = defaultVisibility();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return def;
    const o = JSON.parse(raw) as Partial<Record<EcologySeriesId, boolean>>;
    return { ...def, ...o };
  } catch {
    return def;
  }
}

function saveVisibility(v: Record<EcologySeriesId, boolean>): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(v));
  } catch {
    /* ignore quota / private mode */
  }
}

export class EcologyTrendChart {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private resizeObserver: ResizeObserver | null = null;
  private ticks: number[] = [];
  private orgs: number[] = [];
  private simpson: number[] = [];
  private lineages: number[] = [];
  private topShare: number[] = [];
  private perimeter: number[] = [];
  private drawPending = false;
  private seriesVisible: Record<EcologySeriesId, boolean> = loadVisibility();
  /**
   * Horizontal "time scale": how many recent samples span the plot width.
   * `Infinity` = show the full buffered history (default). Wheel on the chart zooms this in/out.
   */
  private timeWindowSamples = Number.POSITIVE_INFINITY;
  /** When true, the visible window stays locked to the newest samples (right edge). */
  private timeFollowLatest = true;
  /** Index of the first visible sample when not following latest (clamped in `draw`). */
  private timeViewStart = 0;
  private timePanPointerId: number | null = null;
  private timePanLastX = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext('2d', { alpha: false });
    if (!ctx) throw new Error('EcologyTrendChart: 2d context unavailable');
    this.ctx = ctx;
    this.resize();
    if (typeof window !== 'undefined') {
      window.addEventListener('resize', () => this.resize());
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden) this.redrawNow();
      });
    }
    const parent = this.canvas.parentElement;
    if (parent && typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(parent);
    }

    this.mountSeriesToggles();

    this.canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const n = this.ticks.length;
        if (n < 2) return;

        const zoomIn = e.deltaY < 0;
        const factor = zoomIn ? 0.9 : 1.1;

        if (zoomIn) {
          if (!Number.isFinite(this.timeWindowSamples) || this.timeWindowSamples >= n) {
            this.timeWindowSamples = Math.max(MIN_TIME_WINDOW_SAMPLES, Math.floor(n * factor));
          } else {
            this.timeWindowSamples = Math.max(
              MIN_TIME_WINDOW_SAMPLES,
              Math.floor(this.timeWindowSamples * factor),
            );
          }
        } else {
          if (!Number.isFinite(this.timeWindowSamples) || this.timeWindowSamples >= n) {
            return;
          }
          const next = Math.ceil(this.timeWindowSamples * factor);
          if (next >= n) {
            this.timeWindowSamples = Number.POSITIVE_INFINITY;
          } else {
            this.timeWindowSamples = next;
          }
        }
        this.scheduleDraw();
      },
      { passive: false },
    );

    this.canvas.style.touchAction = 'none';
    this.canvas.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      this.timePanPointerId = e.pointerId;
      this.timePanLastX = e.clientX;
      try {
        this.canvas.setPointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      this.canvas.style.cursor = 'grabbing';
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (this.timePanPointerId !== e.pointerId) return;
      const n = this.ticks.length;
      if (n < 2) return;
      const spanAll = this.computeSpanAll(n);
      const plotW = this.plotWidthCss();
      const dx = e.clientX - this.timePanLastX;
      this.timePanLastX = e.clientX;
      const deltaIndex = -Math.round((dx / plotW) * spanAll);
      if (deltaIndex === 0) return;
      this.timeFollowLatest = false;
      const maxStart = Math.max(0, n - spanAll);
      this.timeViewStart = Math.max(0, Math.min(this.timeViewStart + deltaIndex, maxStart));
      this.scheduleDraw();
    });
    const endPan = (e: PointerEvent) => {
      if (this.timePanPointerId !== e.pointerId) return;
      this.timePanPointerId = null;
      try {
        this.canvas.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      this.canvas.style.cursor = '';
      const n = this.ticks.length;
      if (n >= 2) {
        const spanAll = this.computeSpanAll(n);
        const maxStart = Math.max(0, n - spanAll);
        if (this.timeViewStart >= maxStart) {
          this.timeFollowLatest = true;
          this.timeViewStart = maxStart;
        }
      }
    };
    this.canvas.addEventListener('pointerup', endPan);
    this.canvas.addEventListener('pointercancel', endPan);

    this.canvas.addEventListener('mousemove', () => {
      if (this.timePanPointerId !== null) return;
      this.canvas.style.cursor = 'grab';
    });
    this.canvas.addEventListener('mouseleave', () => {
      if (this.timePanPointerId !== null) return;
      this.canvas.style.cursor = '';
    });

    this.canvas.addEventListener('dblclick', () => {
      const n = this.ticks.length;
      if (n < 2) return;
      this.timeFollowLatest = true;
      const spanAll = this.computeSpanAll(n);
      this.timeViewStart = Math.max(0, n - spanAll);
      this.scheduleDraw();
    });
  }

  private mountSeriesToggles(): void {
    const host = document.getElementById('ecology-chart-toggles');
    if (!host) return;
    host.replaceChildren();
    for (const s of ECOLOGY_SERIES) {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = this.seriesVisible[s.id];
      input.addEventListener('change', () => {
        this.seriesVisible[s.id] = input.checked;
        saveVisibility(this.seriesVisible);
        this.scheduleDraw();
      });
      const span = document.createElement('span');
      span.textContent = s.toggleLabel;
      label.append(input, span);
      host.appendChild(label);
    }
  }

  private computeSpanAll(n: number): number {
    return Number.isFinite(this.timeWindowSamples)
      ? Math.max(MIN_TIME_WINDOW_SAMPLES, Math.min(this.timeWindowSamples, n))
      : n;
  }

  private plotWidthCss(): number {
    const w = this.canvas.width / this.dpr;
    return Math.max(1, w - PAD_L - PAD_R);
  }

  private valueAt(id: EcologySeriesId, i: number): number {
    switch (id) {
      case 'orgs':
        return this.orgs[i]!;
      case 'lineages':
        return this.lineages[i]!;
      case 'simpson':
        return this.simpson[i]!;
      case 'topShare':
        return this.topShare[i]!;
      case 'perimeter':
        return this.perimeter[i]!;
    }
  }

  private resize(): void {
    const parent = this.canvas.parentElement;
    let w = parent ? parent.clientWidth : 0;
    if (w < 48) w = 240;
    const cssH = 168;
    const pxr = typeof window !== 'undefined' ? window.devicePixelRatio : 1;
    this.dpr = Math.min(2.5, pxr > 0 && Number.isFinite(pxr) ? pxr : 1);
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${cssH}px`;
    this.canvas.width = Math.max(1, Math.floor(w * this.dpr));
    this.canvas.height = Math.max(1, Math.floor(cssH * this.dpr));
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.scheduleDraw();
  }

  /** Synchronous draw when rAF was deferred (e.g. tab background) so the plot is not left stale. */
  private redrawNow(): void {
    this.drawPending = false;
    this.draw();
  }

  scheduleDraw(): void {
    if (this.drawPending) return;
    this.drawPending = true;
    if (typeof document !== 'undefined' && document.hidden) {
      return;
    }
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
    let shifted = false;
    while (this.ticks.length > MAX_SAMPLES) {
      shifted = true;
      this.ticks.shift();
      this.orgs.shift();
      this.simpson.shift();
      this.lineages.shift();
      this.topShare.shift();
      this.perimeter.shift();
    }
    if (shifted && !this.timeFollowLatest && this.timeViewStart > 0) {
      this.timeViewStart--;
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

    const padL = PAD_L;
    const padR = PAD_R;
    const padT = PAD_T;
    const padB = PAD_B;
    const plotW = Math.max(1, w - padL - padR);
    const plotH = Math.max(1, h - padT - padB);

    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = COL_TEXT;
    ctx.textAlign = 'left';
    ctx.fillText('Ecology & structure', 6, 14);

    const n = this.ticks.length;
    if (n < 2) {
      ctx.fillStyle = COL_TEXT;
      ctx.fillText('Run simulation to plot…', padL, padT + plotH * 0.45);
      return;
    }

    const spanAll = this.computeSpanAll(n);
    const maxStart = Math.max(0, n - spanAll);
    let start: number;
    if (this.timeFollowLatest) {
      start = maxStart;
      this.timeViewStart = start;
    } else {
      start = Math.max(0, Math.min(this.timeViewStart, maxStart));
      this.timeViewStart = start;
    }
    const end = start + spanAll - 1;
    const span = end - start;
    ctx.font = '9px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`${this.ticks[start]}–${this.ticks[end]}`, w - 6, 14);
    ctx.textAlign = 'left';
    ctx.font = '10px ui-monospace, monospace';

    let anyLeft = false;
    let anyRight = false;
    let minO = Infinity;
    let maxO = -Infinity;
    let minL = Infinity;
    let maxL = -Infinity;
    let minR = Infinity;
    let maxR = -Infinity;

    for (let i = start; i <= end; i++) {
      for (const s of ECOLOGY_SERIES) {
        if (!this.seriesVisible[s.id]) continue;
        const v = this.valueAt(s.id, i);
        if (s.axis === 'left') {
          anyLeft = true;
          if (s.id === 'orgs') {
            if (v < minO) minO = v;
            if (v > maxO) maxO = v;
          } else {
            if (v < minL) minL = v;
            if (v > maxL) maxL = v;
          }
        } else {
          anyRight = true;
          if (v < minR) minR = v;
          if (v > maxR) maxR = v;
        }
      }
    }

    if (!anyLeft) {
      minO = 0;
      maxO = 10;
      minL = 0;
      maxL = 2;
    } else {
      minO = Math.min(0, minO);
      maxO = Math.max(10, maxO * 1.08);
      minL = Math.min(0, minL);
      maxL = Math.max(2, maxL * 1.12);
    }
    const leftMax = Math.max(maxO, maxL);
    const leftMin = 0.0;

    if (!anyRight) {
      minR = 0;
      maxR = 1;
    } else {
      minR = Math.max(0, minR - 0.03);
      maxR = Math.min(1, Math.max(maxR + 0.03, minR + 0.06));
    }

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

    const sliceCount = span + 1;
    const m = Math.min(sliceCount, MAX_DRAW_VERTICES);
    const idxAt = (j: number) =>
      start + Math.round((j / Math.max(1, m - 1)) * Math.max(0, sliceCount - 1));
    const xAt = (j: number) => {
      const i = idxAt(j);
      return padL + ((i - start) / span) * plotW;
    };
    const yLeft = (v: number) => padT + plotH - ((v - leftMin) / spanLeft) * plotH;
    const yRight = (v: number) => padT + plotH - ((v - minR) / spanR) * plotH;

    for (const s of ECOLOGY_SERIES) {
      if (!this.seriesVisible[s.id]) continue;
      const yScale = s.axis === 'left' ? yLeft : yRight;
      this.strokeSeries(m, idxAt, xAt, (i) => yScale(this.valueAt(s.id, i)), s.color, s.lineWidth);
    }

    ctx.fillStyle = COL_TEXT;
    ctx.textAlign = 'right';
    ctx.fillText(String(Math.round(leftMax)), padL - 4, padT + 9);
    ctx.fillText(String(Math.round(leftMin)), padL - 4, padT + plotH);
    ctx.textAlign = 'left';
    ctx.fillText(maxR.toFixed(2), padL + plotW + 4, padT + 9);
    ctx.fillText(minR.toFixed(2), padL + plotW + 4, padT + plotH);

    const legY0 = padT + plotH + 8;
    ctx.font = '8px ui-monospace, monospace';
    ctx.textAlign = 'left';

    const leftDefs = ECOLOGY_SERIES.filter((s) => s.axis === 'left' && this.seriesVisible[s.id]);
    const rightDefs = ECOLOGY_SERIES.filter((s) => s.axis === 'right' && this.seriesVisible[s.id]);

    const drawLegendLine = (y: number, prefix: string, defs: readonly EcologySeriesDef[]) => {
      let x = padL;
      ctx.fillStyle = COL_TEXT;
      ctx.fillText(prefix, x, y);
      x += ctx.measureText(prefix).width + 4;
      const dotR = 2.5;
      for (const s of defs) {
        ctx.fillStyle = s.color;
        ctx.beginPath();
        ctx.arc(x + dotR, y - 2.5, dotR, 0, Math.PI * 2);
        ctx.fill();
        x += dotR * 2 + 3;
        ctx.fillStyle = COL_TEXT;
        const t = s.legendShort;
        ctx.fillText(t, x, y);
        x += ctx.measureText(t).width + 8;
      }
    };

    if (leftDefs.length === 0 && rightDefs.length === 0) {
      ctx.fillStyle = COL_TEXT;
      ctx.fillText('No series enabled — use the checkboxes below', padL, legY0);
    } else {
      let ly = legY0;
      if (leftDefs.length > 0) {
        drawLegendLine(ly, 'L count:', leftDefs);
        ly += 11;
      }
      if (rightDefs.length > 0) {
        drawLegendLine(ly, 'R [0-1]:', rightDefs);
      }
    }
  }
}
