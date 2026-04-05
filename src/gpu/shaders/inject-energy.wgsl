struct InjectParams {
  cx: f32,
  cy: f32,
  radius: f32,
  amount: f32,
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: InjectParams;
@group(0) @binding(1) var<storage, read_write> envEnergy: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= params.width || y >= params.height) { return; }

  let dx = f32(x) - params.cx;
  let dy = f32(y) - params.cy;
  let dist = sqrt(dx * dx + dy * dy);

  if (dist < params.radius) {
    let factor = 1.0 - dist / params.radius;
    let idx = y * params.width + x;
    envEnergy[idx] += params.amount * factor;
  }
}
