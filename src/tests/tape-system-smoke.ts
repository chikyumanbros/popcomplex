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

function main() {
  testProtoMoveGate();
  testDataDupProtectionScope();
  testProtoRuleTableSanity();
  testSnapshotRoundtrip();
  console.log('[Tape smoke] OK');
}

main();
