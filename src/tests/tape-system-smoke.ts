import assert from 'node:assert/strict';
import {
  createProtoTape,
  ActionOpcode,
  MAX_RULES,
  CONDITIONS_OFFSET,
  RULE_SIZE,
  MAX_VALID_ACTION_OPCODE,
  tapeSnapshotBase64,
  decodeTapeSnapshotBase64,
  tapeFromSnapshot,
} from '../simulation/tape';
import { isProtectedDataDupTarget } from '../simulation/transcription';
import { World } from '../simulation/world';
import { OrganismManager } from '../simulation/organism';
import { RuleEvaluator } from '../simulation/rule-evaluator';
import { CellType, GRID_WIDTH } from '../simulation/constants';

function idxOf(x: number, y: number): number {
  return y * GRID_WIDTH + x;
}

function collectRuleOpcodes(data: Uint8Array): number[] {
  const out: number[] = [];
  for (let i = 0; i < MAX_RULES; i++) {
    out.push(data[CONDITIONS_OFFSET + i * RULE_SIZE + 2]);
  }
  return out;
}

function testProtoMoveGate() {
  const tape = createProtoTape();
  const divRule = tape.getRule(2);
  const moveRule = tape.getRule(4);
  assert.equal(divRule.actionOpcode, ActionOpcode.DIV, 'rule[2] must be DIV');
  assert.equal(divRule.thresholdValue, 42, 'proto DIV threshold must match tuned baseline');
  assert.equal(moveRule.actionOpcode, ActionOpcode.MOVE, 'rule[4] must be MOVE');
  assert.equal(moveRule.thresholdValue, 144, 'proto MOVE threshold must gate mood confidence');
  assert.equal(moveRule.conditionFlags, 0b00_00_01_00, 'proto MOVE condition encoding changed');
}

function testDataDupProtectionScope() {
  assert.equal(isProtectedDataDupTarget(4), true, 'maxCells slot must be protected');
  assert.equal(isProtectedDataDupTarget(5), false, 'data slot 5 must remain evolvable');
  assert.equal(isProtectedDataDupTarget(6), false, 'data slot 6 must remain evolvable');
}

function testProtoRuleTableSanity() {
  const tape = createProtoTape();
  for (let i = 0; i < MAX_RULES; i++) {
    const rule = tape.getRule(i);
    const off = CONDITIONS_OFFSET + i * RULE_SIZE;
    const rawOpcode = tape.data[off + 2];

    assert.ok(rawOpcode <= MAX_VALID_ACTION_OPCODE, `proto raw opcode out of range at rule ${i}`);
    assert.ok(rule.actionParam >= 0 && rule.actionParam < 32, `proto actionParam must target data node at rule ${i}`);
  }
}

function testSnapshotRoundtrip() {
  const src = createProtoTape();
  src.degradation[0] = 17;
  src.degradation[7] = 99;
  src.degradation[255] = 5;

  const b64 = tapeSnapshotBase64(src);
  const decoded = decodeTapeSnapshotBase64(b64);
  assert.ok(decoded, 'snapshot decode must succeed');
  const rebuilt = tapeFromSnapshot(decoded!.data, decoded!.degradation);

  assert.deepEqual([...rebuilt.data], [...src.data], 'snapshot roundtrip must preserve tape data bytes');
  assert.deepEqual(
    [...rebuilt.degradation],
    [...src.degradation],
    'snapshot roundtrip must preserve degradation bytes',
  );
}

function testSpillLocalRedistribution() {
  const world = new World();
  const organisms = new OrganismManager();
  const tape = createProtoTape();
  const orgId = 1;
  organisms.register(orgId, tape);
  const x = 10;
  const y = 10;
  const idx = idxOf(x, y);
  world.setCell(x, y, orgId, CellType.Stem, 20, tape.getLineagePacked());
  organisms.get(orgId)!.cells.add(idx);
  world.setStomachByIdx(idx, 1.0);

  const evaluator = new RuleEvaluator(world, organisms);
  const cell = { x, y, idx, energy: world.getCellEnergyByIdx(idx), orgId };
  const ok = (evaluator as any).actionSpill(cell, 0.8);
  assert.equal(ok, true, 'SPILL must succeed when stomach has enough content');

  const stomachAfter = world.getStomachByIdx(idx);
  assert.ok(stomachAfter < 1.0, 'SPILL must consume own stomach');
  assert.ok((evaluator as any).envEnergy[idx] > 0, 'SPILL must add energy at self cell');
  const northIdx = idxOf(x, y - 1);
  assert.ok((evaluator as any).envEnergy[northIdx] > 0, 'SPILL must distribute some energy to orthogonal neighbors');
}

function testJamBlocksCrossLineageTransferAndAbsorbCoupling() {
  const world = new World();
  const organisms = new OrganismManager();
  const hostTape = createProtoTape();
  const donorTape = createProtoTape();
  for (let i = 0; i < MAX_RULES; i++) {
    const off = CONDITIONS_OFFSET + i * RULE_SIZE + 2;
    hostTape.data[off] = ActionOpcode.NOP;
    donorTape.data[off] = ActionOpcode.REPAIR;
  }

  const hostId = 1;
  const donorId = 2;
  organisms.register(hostId, hostTape);
  organisms.register(donorId, donorTape);

  const hx = 20;
  const hy = 20;
  const dx = 21;
  const dy = 20;
  const hostIdx = idxOf(hx, hy);
  const donorIdx = idxOf(dx, dy);
  world.setCell(hx, hy, hostId, CellType.Stem, 6, 0x123456);
  world.setCell(dx, dy, donorId, CellType.Stem, 8, 0x123456);
  world.setStomachByIdx(hostIdx, 10);
  world.setMorphogenA(hostIdx, 2);
  world.setMorphogenB(hostIdx, 2);
  world.setMorphogenA(donorIdx, 2);
  world.setMorphogenB(donorIdx, 2);
  organisms.get(hostId)!.cells.add(hostIdx);
  organisms.get(donorId)!.cells.add(donorIdx);

  const evaluator = new RuleEvaluator(world, organisms);
  const hostCell = { x: hx, y: hy, idx: hostIdx, energy: world.getCellEnergyByIdx(hostIdx), orgId: hostId };

  const jamOk = (evaluator as any).actionJam(hostCell, 1.0);
  assert.equal(jamOk, true, 'JAM must activate on foreign boundary');

  const beforeOpcodes = collectRuleOpcodes(hostTape.data);
  (evaluator as any).tryHorizontalTapeTransfer(hostCell, donorIdx, donorId, 1.0, 1.0);
  const afterOpcodes = collectRuleOpcodes(hostTape.data);
  assert.deepEqual(afterOpcodes, beforeOpcodes, 'JAM must block cross-lineage opcode transfer while active');

  const stomachBefore = world.getStomachByIdx(hostIdx);
  const absorbOk = (evaluator as any).actionAbsorb(hostCell, 2.0);
  assert.equal(absorbOk, true, 'ABSORB should still be able to execute under JAM');
  assert.ok(world.getStomachByIdx(hostIdx) > stomachBefore, 'JAM must suppress compatible absorb coupling route and fall back to one-way absorb');
}

function main() {
  testProtoMoveGate();
  testDataDupProtectionScope();
  testProtoRuleTableSanity();
  testSnapshotRoundtrip();
  testSpillLocalRedistribution();
  testJamBlocksCrossLineageTransferAndAbsorbCoupling();
  console.log('[Tape smoke] OK');
}

main();
