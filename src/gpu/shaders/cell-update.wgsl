struct Uniforms {
  width: u32,
  height: u32,
  tick: u32,
  pingpong: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> cellIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellOut: array<u32>;
@group(0) @binding(3) var<storage, read_write> envEnergy: array<f32>;

// Passive energy leak from cells with no valid rules (dysfunctional)
const LEAK_RATE: f32 = 0.1;

fn getCellType(base: u32) -> u32 {
  return cellIn[base + 1u] & 0xFFu;
}

fn getOrgId(base: u32) -> u32 {
  return cellIn[base];
}

fn getCellEnergy(base: u32) -> f32 {
  return bitcast<f32>(cellIn[base + 2u]);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= u.width || y >= u.height) { return; }

  let index = y * u.width + x;
  let base = index * 8u;

  // Copy current state forward
  for (var i = 0u; i < 8u; i++) {
    cellOut[base + i] = cellIn[base + i];
  }

  let cellType = getCellType(base);
  let energy = getCellEnergy(base);
  let orgId = getOrgId(base);

  if (cellType == 0u) { return; }

  // Cells with very low energy leak to environment and dissolve
  if (energy < 0.5 && orgId > 0u) {
    let leak = min(energy, LEAK_RATE);
    let newEnergy = energy - leak;
    cellOut[base + 2u] = bitcast<u32>(max(newEnergy, 0.0));
    envEnergy[index] += leak;

    if (newEnergy <= 0.01) {
      // Cell dissolves: return remaining energy to environment
      envEnergy[index] += max(newEnergy, 0.0);
      cellOut[base] = 0u;
      cellOut[base + 1u] = 0u;
      cellOut[base + 2u] = 0u;
      cellOut[base + 3u] = 0u;
    }
  }

  // Increment age
  let ageFlags = cellIn[base + 5u];
  let age = (ageFlags & 0xFFFFu) + 1u;
  let flags = ageFlags >> 16u;
  cellOut[base + 5u] = (flags << 16u) | min(age, 0xFFFFu);
}
