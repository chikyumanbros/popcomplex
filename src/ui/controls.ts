import { GRID_WIDTH, GRID_HEIGHT } from '../simulation/constants';

/** Fragment shader visualization mode (`render.wgsl`); cycle with V. */
export const VIEW_MODE_COUNT = 6;
export const VIEW_MODE_NAMES = [
  'Default',
  'Env heat',
  'Morphogens',
  'Stomach',
  'Markers',
  'Energy',
] as const;

export interface UIState {
  paused: boolean;
  speed: number;
  mouseGridX: number;
  mouseGridY: number;
  viewX: number;
  viewY: number;
  viewZoom: number;
  viewMode: number;
  stepRequested: boolean;
}

/** Sync all view-mode labels (header pill, sidebar banner, stats row). Call each frame from `updateStats`. */
export function setViewModeUi(viewMode: number) {
  const m = ((viewMode % VIEW_MODE_COUNT) + VIEW_MODE_COUNT) % VIEW_MODE_COUNT;
  const name = VIEW_MODE_NAMES[m] ?? '?';
  const idx = `${m + 1}/${VIEW_MODE_COUNT}`;
  const hdr = document.getElementById('header-view-mode');
  if (hdr) {
    hdr.textContent = `View ${idx} — ${name}`;
  }
  const banVal = document.getElementById('view-mode-banner-val');
  if (banVal) banVal.textContent = name;
  const banHint = document.getElementById('view-mode-banner-hint');
  if (banHint) banHint.textContent = `${idx} · V next`;
  setStat('stat-view', `${name} (${idx})`);
}

/** Same mapping as `render.wgsl` viewUV → cell index (screen px → grid). */
export function pointerClientToGrid(
  canvas: HTMLCanvasElement,
  ui: UIState,
  clientX: number,
  clientY: number,
): { gx: number; gy: number } {
  const rect = canvas.getBoundingClientRect();
  const normX = (clientX - rect.left) / rect.width;
  const normY = (clientY - rect.top) / rect.height;
  const worldX = (normX - 0.5) / ui.viewZoom + ui.viewX / GRID_WIDTH + 0.5;
  const worldY = (normY - 0.5) / ui.viewZoom + ui.viewY / GRID_HEIGHT + 0.5;
  return {
    gx: Math.floor(worldX * GRID_WIDTH),
    gy: Math.floor(worldY * GRID_HEIGHT),
  };
}

/** Grid cell at the center of the current view (matches fragment shader uv = 0.5,0.5). */
export function viewPortCenterGrid(ui: UIState): { gx: number; gy: number } {
  const viewUVx = ui.viewX / GRID_WIDTH + 0.5;
  const viewUVy = ui.viewY / GRID_HEIGHT + 0.5;
  return {
    gx: Math.floor(viewUVx * GRID_WIDTH),
    gy: Math.floor(viewUVy * GRID_HEIGHT),
  };
}

function clientToGrid(
  canvas: HTMLCanvasElement,
  state: UIState,
  clientX: number,
  clientY: number,
) {
  const { gx, gy } = pointerClientToGrid(canvas, state, clientX, clientY);
  state.mouseGridX = gx;
  state.mouseGridY = gy;
}

export function createUI(canvas: HTMLCanvasElement): UIState {
  const state: UIState = {
    paused: true,
    speed: 1,
    // Default to world center until the pointer moves on the canvas
    mouseGridX: Math.floor(GRID_WIDTH / 2),
    mouseGridY: Math.floor(GRID_HEIGHT / 2),
    viewX: 0,
    viewY: 0,
    viewZoom: 1,
    viewMode: 0,
    stepRequested: false,
  };

  const panel = document.getElementById('controls')!;
  panel.innerHTML = `
    <div class="control-row">
      <button id="btn-pause">Resume</button>
      <button id="btn-step">Step</button>
      <label>Speed: <input id="speed" type="range" min="1" max="20" value="1"><span id="speed-val"> 1</span></label>
    </div>
    <div class="control-row">
      <button id="btn-spawn">Spawn life</button>
      <button id="btn-restart" type="button" title="Reload page; keeps current URL query (same seed / multiOrigin / etc.)">Restart</button>
      <button id="btn-ai-handoff" type="button" title="Copy markdown report for AI / debugging">Copy AI report</button>
    </div>
    <div class="control-row">
      <label>Prompt:
        <select id="ai-prompt-preset" title="Preset appended by 'Copy AI report + prompt'">
          <option value="review">General review</option>
          <option value="ecology">Ecology focus</option>
          <option value="tape">Tape health</option>
        </select>
      </label>
      <button id="btn-ai-handoff-prompt" type="button" title="Copy report plus a ready-to-paste AI prompt">Copy AI report + prompt</button>
    </div>
    <div class="control-row hint">
      Click: spawn at cell · Shift+click: inspect<br>
      Spawn button: view center · Scroll: zoom · Middle-drag: pan<br>
      <kbd>V</kbd>: cycle field view (default → env → morph → stomach → markers → energy)
    </div>
    <div class="view-mode-banner" title="Press V to cycle — same as top-right header pill">
      <span class="view-mode-banner-label">Field view</span>
      <span class="view-mode-banner-val" id="view-mode-banner-val">Default</span>
      <span class="view-mode-banner-hint" id="view-mode-banner-hint">1/6 · V next</span>
    </div>
    <div class="stats-panel">
      <div class="stats-kv stats-kv-main"><span class="stat-lbl">Tick</span><span class="stat-val" id="stat-tick"></span></div>
      <div class="stats-kv stats-kv-main"><span class="stat-lbl">Org</span><span class="stat-val" id="stat-org"></span></div>
      <div class="stats-kv stats-kv-main"><span class="stat-lbl">FPS</span><span class="stat-val" id="stat-fps"></span></div>
      <div class="stats-kv stats-kv-main"><span class="stat-lbl">View</span><span class="stat-val" id="stat-view">Default (1/6)</span></div>
      <div class="stats-kv stats-kv-energy"><span class="stat-lbl">Budget</span><span class="stat-val" id="stat-budget"></span></div>
      <div class="stats-kv stats-kv-energy"><span class="stat-lbl">Meas</span><span class="stat-val" id="stat-meas"></span></div>
      <div class="stats-kv stats-kv-energy"><span class="stat-lbl">Δ</span><span class="stat-val" id="stat-drift"></span></div>
    </div>
  `;

  document.getElementById('btn-pause')!.addEventListener('click', () => {
    state.paused = !state.paused;
    document.getElementById('btn-pause')!.textContent = state.paused ? 'Resume' : 'Pause';
  });

  document.getElementById('btn-step')!.addEventListener('click', () => {
    state.stepRequested = true;
  });

  document.getElementById('speed')!.addEventListener('input', (e) => {
    state.speed = parseInt((e.target as HTMLInputElement).value);
    document.getElementById('speed-val')!.textContent = String(state.speed).padStart(2, ' ');
  });

  document.getElementById('btn-restart')!.addEventListener('click', () => {
    window.location.reload();
  });

  // Mouse grid tracks pointer; also refreshed on zoom/pan so it stays aligned without wiggling the mouse
  canvas.addEventListener('mousemove', (e) => {
    clientToGrid(canvas, state, e.clientX, e.clientY);
  });

  // Zoom with scroll
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    state.viewZoom = Math.max(0.5, Math.min(20, state.viewZoom * zoomFactor));
    clientToGrid(canvas, state, e.clientX, e.clientY);
  }, { passive: false });

  // Pan with middle mouse button
  let isPanning = false;
  canvas.addEventListener('mousedown', (e) => {
    if (e.button === 1) { isPanning = true; e.preventDefault(); }
  });
  canvas.addEventListener('mouseup', (e) => {
    if (e.button === 1) isPanning = false;
  });
  canvas.addEventListener('mousemove', (e) => {
    if (isPanning) {
      state.viewX -= e.movementX / state.viewZoom;
      state.viewY -= e.movementY / state.viewZoom;
      clientToGrid(canvas, state, e.clientX, e.clientY);
    }
  });

  window.addEventListener('keydown', (e) => {
    if (e.key !== 'v' && e.key !== 'V') return;
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    e.preventDefault();
    state.viewMode = (state.viewMode + 1) % VIEW_MODE_COUNT;
    setViewModeUi(state.viewMode);
  });

  setViewModeUi(state.viewMode);

  return state;
}

function setStat(id: string, text: string) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

/** Updates one row per stat — narrow sidebar safe, values share a right column width. */
export function updateStats(
  tick: number,
  orgCount: number,
  fps: number,
  closed: number,
  measured: number,
  drift: number,
  viewMode: number,
) {
  setStat('stat-tick', String(tick));
  setStat('stat-org', String(orgCount));
  const fpsStr = fps.toFixed(0);
  setStat('stat-fps', fpsStr);
  const fpsHdr = document.getElementById('fps');
  if (fpsHdr) fpsHdr.textContent = `${fpsStr} fps`;
  setStat('stat-budget', `${(closed / 1e6).toFixed(3)}M`);
  setStat('stat-meas', `${(measured / 1e6).toFixed(3)}M`);
  setStat('stat-drift', drift >= 0 ? `+${drift.toFixed(2)}` : drift.toFixed(2));
  setViewModeUi(viewMode);
}
