// -----------------------------------------------------------------------------
// NOT USED by the live app: `main.ts` never dispatches this compute pass. Neural
// dynamics are advanced on the CPU in `RuleEvaluator.propagateSignals`. Keep this
// file in sync with that logic only if you intentionally move stepping to GPU.
// -----------------------------------------------------------------------------

struct Uniforms {
  width: u32,
  height: u32,
  tick: u32,
  pingpong: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> cellIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellOut: array<u32>;

// Neural states: 0=Resting, 1=Excited, 2=Refractory
const FIRE_THRESHOLD: f32 = 0.5;
const REFRACTORY_PERIOD: u32 = 3u;
const SIGNAL_DECAY: f32 = 0.8;

fn getOrgId(base: u32) -> u32 { return cellIn[base]; }
fn getCellType(base: u32) -> u32 { return cellIn[base + 1u] & 0xFFu; }
fn getNeuralState(base: u32) -> u32 { return (cellIn[base + 1u] >> 8u) & 0xFFu; }
fn getRefractoryCnt(base: u32) -> u32 { return (cellIn[base + 1u] >> 16u) & 0xFFu; }
fn getSignalOut(base: u32) -> f32 { return bitcast<f32>(cellIn[base + 3u]); }

fn cellBase(x: u32, y: u32) -> u32 {
  return (y * u.width + x) * 8u;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= u.width || y >= u.height) { return; }

  let index = y * u.width + x;
  let base = index * 8u;

  // Copy state forward
  for (var i = 0u; i < 8u; i++) {
    cellOut[base + i] = cellIn[base + i];
  }

  let cellType = getCellType(base);
  if (cellType == 0u) { return; }

  let orgId = getOrgId(base);
  let neuralState = getNeuralState(base);
  let refCnt = getRefractoryCnt(base);

  var newNeuralState = neuralState;
  var newRefCnt = refCnt;
  var newSignal: f32 = 0.0;

  if (neuralState == 0u) {
    // Resting: accumulate input from same-organism neighbors
    var input: f32 = 0.0;
    if (x > 0u)            { let nb = cellBase(x - 1u, y); if (getOrgId(nb) == orgId) { input += getSignalOut(nb); } }
    if (x < u.width - 1u)  { let nb = cellBase(x + 1u, y); if (getOrgId(nb) == orgId) { input += getSignalOut(nb); } }
    if (y > 0u)            { let nb = cellBase(x, y - 1u); if (getOrgId(nb) == orgId) { input += getSignalOut(nb); } }
    if (y < u.height - 1u) { let nb = cellBase(x, y + 1u); if (getOrgId(nb) == orgId) { input += getSignalOut(nb); } }

    if (input > FIRE_THRESHOLD) {
      newNeuralState = 1u; // -> Excited
      newSignal = 1.0;
    }
  } else if (neuralState == 1u) {
    // Excited -> Refractory
    newNeuralState = 2u;
    newRefCnt = REFRACTORY_PERIOD;
    newSignal = 0.0;
  } else {
    // Refractory countdown
    if (refCnt <= 1u) {
      newNeuralState = 0u;
      newRefCnt = 0u;
    } else {
      newRefCnt = refCnt - 1u;
    }
    newSignal = 0.0;
  }

  // Pack neural_state(8) | refractory_cnt(8) back
  let packed = (cellType) | (newNeuralState << 8u) | (newRefCnt << 16u);
  cellOut[base + 1u] = packed;
  cellOut[base + 3u] = bitcast<u32>(newSignal * SIGNAL_DECAY);
}
