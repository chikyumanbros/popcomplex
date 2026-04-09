/**
 * OrgPortrait — floating popup that renders a Lenia "portrait" of a selected organism.
 *
 * Lifecycle:
 *   open(org, world)  — seeds the Lenia engine from org data, shows the overlay
 *   tick(organisms)   — called each simulation frame; updates NN color state.
 *                       When the org dies the portrait stays open (Lenia keeps running)
 *                       and shows a "dissolved" badge — user closes manually.
 *   close()           — hides overlay, stops animation loop
 */

import type { Organism, OrganismManager } from '../simulation/organism';
import type { World } from '../simulation/world';
import { deriveLeniaParams, LeniaEngine, LENIA_W, LENIA_H } from './lenia-sim';
import { tapeDegradationPercent } from '../simulation/tape-health';

const MOOD_LABELS = ['EAT', 'GROW', 'MOVE', 'SAVE'] as const;
const STEPS_PER_FRAME = 2; // Lenia steps advanced per animation frame

export class OrgPortrait {
  private overlay: HTMLDivElement;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private imageData: ImageData;
  private engine: LeniaEngine | null = null;
  private currentOrgId: number | null = null;
  private rafId = 0;
  /** True once the tracked org is confirmed dead (portrait keeps running freely). */
  private dissolved = false;

  // Cached for per-frame color updates (no field re-seed needed)
  private kinTag = 0;
  private nnDominant = 0;
  private nnPrimitives: Float32Array = new Float32Array(4);
  private nnOutput: Float32Array = new Float32Array(4);

  constructor() {
    this.overlay = document.createElement('div');
    this.overlay.className = 'org-portrait-overlay';
    this.overlay.style.display = 'none';

    const header = document.createElement('div');
    header.className = 'op-header';

    const title = document.createElement('span');
    title.className = 'op-title';
    title.textContent = 'Portrait';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'op-close';
    closeBtn.textContent = '×';
    closeBtn.setAttribute('aria-label', 'Close portrait');
    closeBtn.addEventListener('click', () => this.close());

    header.appendChild(title);
    header.appendChild(closeBtn);

    this.canvas = document.createElement('canvas');
    this.canvas.width = LENIA_W;
    this.canvas.height = LENIA_H;
    this.canvas.className = 'op-canvas';

    const stats = document.createElement('div');
    stats.className = 'op-stats';

    const moodBar = document.createElement('div');
    moodBar.className = 'op-mood-bar';

    const tapeBar = document.createElement('div');
    tapeBar.className = 'op-tape-bar';

    this.overlay.appendChild(header);
    this.overlay.appendChild(this.canvas);
    this.overlay.appendChild(stats);
    this.overlay.appendChild(moodBar);
    this.overlay.appendChild(tapeBar);
    document.body.appendChild(this.overlay);

    const c = this.canvas.getContext('2d');
    if (!c) throw new Error('Could not get 2D context for portrait canvas');
    this.ctx = c;
    this.imageData = this.ctx.createImageData(LENIA_W, LENIA_H);

    // Allow dragging the overlay
    this.initDrag(header);
  }

  /** Open (or replace) portrait for the given organism. */
  open(org: Organism, world: World): void {
    // Stop any existing animation before replacing.
    if (this.rafId !== 0) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }

    this.currentOrgId = org.id;
    this.dissolved = false;

    // Derive Lenia params from tape.
    const params = deriveLeniaParams(org.tape.data);
    this.engine = new LeniaEngine(params);
    this.engine.seedFromOrganism(org, world);

    // Grab color state.
    this.kinTag = org.tape.getPublicKinTagPacked() & 0xffffff;
    this.nnDominant = org.nnDominant;
    this.nnPrimitives = new Float32Array(org.nnPrimitives);
    this.nnOutput = new Float32Array(org.nnOutput);

    this.updateHeader(org);
    this.updateStats(org);
    this.updateMoodBar(org);
    this.updateTapeBar(org);

    this.overlay.style.display = 'flex';
    this.overlay.classList.remove('op-dissolved');

    // Start animation loop.
    this.rafId = requestAnimationFrame(this.animate);
  }

  /** Update NN-derived color state each simulation frame (while org is alive). */
  updateNNState(org: Organism): void {
    if (org.id !== this.currentOrgId || this.dissolved) return;
    this.nnDominant = org.nnDominant;
    this.nnPrimitives.set(org.nnPrimitives);
    this.nnOutput.set(org.nnOutput);
    this.updateMoodBar(org);
  }

  /**
   * Call each main loop frame.
   * If the org has died, marks portrait as dissolved but keeps Lenia running —
   * the user must click × to close.
   */
  tick(organisms: OrganismManager): void {
    if (this.currentOrgId === null) return;
    if (this.dissolved) return; // already marked, nothing more to update

    const org = organisms.get(this.currentOrgId);
    if (!org) {
      // Org died — show dissolved badge but keep portrait alive.
      this.dissolved = true;
      this.overlay.classList.add('op-dissolved');
      const title = this.overlay.querySelector('.op-title') as HTMLSpanElement | null;
      if (title) title.textContent = `Org #${this.currentOrgId}  †`;
      const stats = this.overlay.querySelector('.op-stats') as HTMLDivElement | null;
      if (stats) stats.textContent = 'dissolved';
      return;
    }
    this.updateStats(org);
    this.updateHeader(org);
  }

  close(): void {
    this.overlay.style.display = 'none';
    this.overlay.classList.remove('op-dissolved');
    this.currentOrgId = null;
    this.dissolved = false;
    this.engine = null;
    if (this.rafId !== 0) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
  }

  get isOpen(): boolean {
    return this.currentOrgId !== null;
  }

  // ---- private helpers ----

  private animate = (): void => {
    // Stop only if engine was destroyed (explicit close).
    if (!this.engine) {
      this.rafId = 0;
      return;
    }

    for (let i = 0; i < STEPS_PER_FRAME; i++) this.engine.step();

    this.engine.renderToImageData(this.imageData, this.kinTag, this.nnDominant, this.nnPrimitives);
    this.ctx.putImageData(this.imageData, 0, 0);

    this.rafId = requestAnimationFrame(this.animate);
  };

  private updateHeader(org: Organism): void {
    const title = this.overlay.querySelector('.op-title') as HTMLSpanElement;
    if (title) title.textContent = `Org #${org.id}  ·  age ${org.age}`;
  }

  private updateStats(org: Organism): void {
    const stats = this.overlay.querySelector('.op-stats') as HTMLDivElement | null;
    if (!stats) return;

    // Compute average energy across all cells.
    let totalE = 0;
    let count = 0;
    // nnStress as proxy for "energy" since we don't have a world ref here.
    // We use nnInput[0] (energy-ish) but keep it simple: show cells + nnStress.
    count = org.cells.size;

    stats.textContent =
      `cells ${String(count).padStart(3, '\u2007')}  ` +
      `stress ${org.nnStress01.toFixed(2)}  ` +
      `claim ${org.nnClaim01.toFixed(2)}`;
    void totalE; // suppress unused
  }

  private updateMoodBar(org: Organism): void {
    const bar = this.overlay.querySelector('.op-mood-bar') as HTMLDivElement | null;
    if (!bar) return;

    const dominant = org.nnDominant;
    const scores = org.nnOutput;
    let html = '<span class="op-mood-label">mood</span>';
    for (let i = 0; i < 4; i++) {
      const pct = Math.round(scores[i]! * 100);
      const active = i === dominant ? ' op-mood-active' : '';
      html += `<span class="op-mood-seg${active}">${MOOD_LABELS[i]} ${pct}%</span>`;
    }
    bar.innerHTML = html;
  }

  private updateTapeBar(org: Organism): void {
    const bar = this.overlay.querySelector('.op-tape-bar') as HTMLDivElement | null;
    if (!bar) return;

    const deg = tapeDegradationPercent(org.tape.degradation);
    const health = (100 - deg).toFixed(0);
    const filled = Math.round((100 - deg) / 5); // 0..20 segments
    const segs = '█'.repeat(filled) + '░'.repeat(20 - filled);
    bar.innerHTML = `<span class="op-tape-label">tape</span><span class="op-tape-segs">${segs}</span><span class="op-tape-pct">${health}%</span>`;
  }

  private initDrag(handle: HTMLElement): void {
    let dragging = false;
    let startX = 0, startY = 0, startLeft = 0, startTop = 0;

    handle.style.cursor = 'grab';

    handle.addEventListener('mousedown', (e) => {
      dragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = this.overlay.getBoundingClientRect();
      startLeft = rect.left;
      startTop = rect.top;
      handle.style.cursor = 'grabbing';
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      this.overlay.style.left = `${startLeft + dx}px`;
      this.overlay.style.top = `${startTop + dy}px`;
      this.overlay.style.right = 'auto';
      this.overlay.style.bottom = 'auto';
    });

    document.addEventListener('mouseup', () => {
      dragging = false;
      handle.style.cursor = 'grab';
    });
  }
}
