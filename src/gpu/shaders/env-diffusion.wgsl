struct Uniforms {
  width: u32,
  height: u32,
  tick: u32,
  pingpong: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> envIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> envOut: array<f32>;

const DIFFUSION_RATE: f32 = 0.05;

fn idx(x: u32, y: u32) -> u32 {
  return y * u.width + x;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= u.width || y >= u.height) { return; }

  let i = idx(x, y);
  let center = envIn[i];

  var neighborSum: f32 = 0.0;
  var count: f32 = 0.0;

  for (var dy: i32 = -1; dy <= 1; dy++) {
    for (var dx: i32 = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let nx = i32(x) + dx;
      let ny = i32(y) + dy;
      if (nx < 0 || ny < 0 || nx >= i32(u.width) || ny >= i32(u.height)) { continue; }
      neighborSum += envIn[idx(u32(nx), u32(ny))];
      count += 1.0;
    }
  }

  let avg = neighborSum / count;
  let diff = (avg - center) * DIFFUSION_RATE;
  envOut[i] = center + diff;
}
