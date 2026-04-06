import type { OrganismManager } from '../simulation/organism';
import type { World } from '../simulation/world';
import { U32_PER_CELL } from '../simulation/world';
import type { UIState } from './controls';
import { GRID_WIDTH, GRID_HEIGHT } from '../simulation/constants';
import { formatTapeRulesInspectorHtml } from '../simulation/tape';

export function setupInspector(
  canvas: HTMLCanvasElement,
  world: World,
  organisms: OrganismManager,
  ui: UIState,
) {
  const infoEl = document.getElementById('inspector')!;

  canvas.addEventListener('click', (e) => {
    if (!e.shiftKey) return;

    const gx = ui.mouseGridX;
    const gy = ui.mouseGridY;

    const gxP = String(gx).padStart(3, '\u2007');
    const gyP = String(gy).padStart(3, '\u2007');
    const coord = `${gxP},${gyP}`;

    if (gx < 0 || gx >= GRID_WIDTH || gy < 0 || gy >= GRID_HEIGHT) {
      infoEl.innerHTML = `<span class="dim">Out of bounds (${coord})</span>`;
      return;
    }

    const orgId = world.getOrganismId(gx, gy);
    if (orgId === 0) {
      infoEl.innerHTML = `<span class="dim">(${coord}) empty</span>`;
      return;
    }

    const org = organisms.get(orgId);
    if (!org) {
      infoEl.innerHTML = `<span class="dim">(${coord}) orphaned org #${orgId}</span>`;
      return;
    }

    const idx = gy * GRID_WIDTH + gx;
    const energy = world.getCellEnergy(gx, gy).toFixed(1).padStart(6, '\u2007');
    const stomach = world.getStomachByIdx(idx).toFixed(1).padStart(6, '\u2007');
    const morphA = world.getMorphogenA(idx).toFixed(2).padStart(5, '\u2007');
    const morphB = world.getMorphogenB(idx).toFixed(2).padStart(5, '\u2007');
    const [mEat, mDigest, mSignal, mMove] = world.getMarkersByIdx(idx);
    const kinFaceTape = org.tape.getPublicKinTagPacked() & 0xffffff;
    const kinGeneticTape = org.tape.getGeneticKinTagPacked() & 0xffffff;
    const lineageCell = world.cellData[idx * U32_PER_CELL + 7] & 0xffffff;
    const degradation = org.tape.degradation.reduce((a: number, b: number) => a + b, 0);
    const maxDeg = org.tape.degradation.length * 255;
    const degPercent = ((degradation / maxDeg) * 100).toFixed(1).padStart(5, '\u2007');

    const nnOut = org.nnOutput;
    const moodLabels = ['eat', 'grow', 'move', 'save'];
    const nnStr = moodLabels
      .map((m, i) => `${m}:${nnOut[i].toFixed(2).padStart(5, '\u2007')}`)
      .join(' ');

    let tapeHtml = '<div class="tape-view">';
    const bpl = 16;
    // Color thresholds: keep early “normal wear” uncolored; highlight only when wear likely matters for survival.
    const WEAR_WARN = 32;
    const WEAR_DANGER = 96;
    const regionLabels: Record<number, string> = { 0: '── data ──', 32: '── CA ──', 64: '── rules ──', 128: '── NN wt ──' };
    for (let off = 0; off < org.tape.data.length; off += bpl) {
      if (regionLabels[off]) tapeHtml += `<div style="color:#555">${regionLabels[off]}</div>`;
      const hex: string[] = [];
      for (let b = 0; b < bpl && off + b < org.tape.data.length; b++) {
        const i = off + b;
        const h = org.tape.data[i].toString(16).padStart(2, '0');
        const d = org.tape.degradation[i];
        hex.push(d > WEAR_DANGER ? `<span class="corrupt-high">${h}</span>` : d > WEAR_WARN ? `<span class="corrupt-low">${h}</span>` : h);
      }
      tapeHtml += `<div>${off.toString(16).padStart(3, '0')}: ${hex.join(' ')}</div>`;
    }
    tapeHtml += '</div>';

    const ageS = String(org.age).padStart(7, '\u2007');
    const cellsS = String(org.cells.size).padStart(4, '\u2007');
    const maxCellS = String(org.tape.getMaxCells()).padStart(4, '\u2007');
    const mEatS = String(mEat).padStart(3, '\u2007');
    const mDigestS = String(mDigest).padStart(3, '\u2007');
    const mSignalS = String(mSignal).padStart(3, '\u2007');
    const mMoveS = String(mMove).padStart(3, '\u2007');

    infoEl.innerHTML = `
      <div class="inspector-header">Org #${orgId}  (${coord})</div>
      <table class="inspector-table">
        <tr><td>Age</td><td class="ins-num">${ageS}</td><td>Cells</td><td class="ins-num">${cellsS}</td></tr>
        <tr><td>Energy</td><td class="ins-num">${energy}</td><td>Stomach</td><td class="ins-num">${stomach}</td></tr>
        <tr><td>MorphA</td><td class="ins-num">${morphA}</td><td>MorphB</td><td class="ins-num">${morphB}</td></tr>
        <tr><td>Tape deg.</td><td class="ins-num">${degPercent}%</td><td>MaxCells</td><td class="ins-num">${maxCellS}</td></tr>
        <tr><td colspan="4" class="ins-lineage">Kin face #${kinFaceTape.toString(16).padStart(6, '0')} · genetic #${kinGeneticTape.toString(16).padStart(6, '0')} · cell #${lineageCell.toString(16).padStart(6, '0')}</td></tr>
      </table>
      <div class="ins-markers">
        Markers  eat${mEatS}  dig${mDigestS}  sig${mSignalS}  move${mMoveS}
      </div>
      <div class="nn-bias">NN  ${nnStr}</div>
      ${formatTapeRulesInspectorHtml(org.tape)}
      ${tapeHtml}
    `;
  });
}
