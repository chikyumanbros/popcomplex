struct Uniforms {
  width: u32,
  height: u32,
  tick: u32,
  pingpong: u32,
  viewX: f32,
  viewY: f32,
  viewZoom: f32,
  viewMode: u32,
};

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> cellState: array<u32>;
@group(0) @binding(2) var<storage, read> envEnergy: array<f32>;
@group(0) @binding(3) var<storage, read> componentMask: array<u32>;

fn getCellType(index: u32) -> u32 {
  return cellState[index * 8u + 1u] & 0xFFu;
}

fn getNeuralState(index: u32) -> u32 {
  return (cellState[index * 8u + 1u] >> 8u) & 0xFFu;
}

fn getCellEnergy(index: u32) -> f32 {
  return bitcast<f32>(cellState[index * 8u + 2u]);
}

fn getOrgId(index: u32) -> u32 {
  return cellState[index * 8u];
}

fn getStomach(index: u32) -> f32 {
  return bitcast<f32>(cellState[index * 8u + 4u]);
}

fn getMorphA(index: u32) -> f32 {
  return bitcast<f32>(cellState[index * 8u + 5u]);
}

fn getMorphB(index: u32) -> f32 {
  return bitcast<f32>(cellState[index * 8u + 6u]);
}

fn getMarkerRgb(index: u32) -> vec3f {
  let v = cellState[index * 8u + 3u];
  let r = f32(v & 0xFFu);
  let g = f32((v >> 8u) & 0xFFu);
  let b = f32((v >> 16u) & 0xFFu);
  return vec3f(r, g, b) / 255.0;
}

// Organism hues: slightly richer saturation than env for readable clades
fn hashToMutedRgb(hash: u32) -> vec3f {
  let h = f32(hash % 1000u) / 1000.0;
  let s = 0.40 + f32((hash >> 10u) % 200u) / 1000.0;
  let v = 0.82 + f32((hash >> 20u) % 100u) / 1000.0;
  let c = v * s;
  let x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0));
  let m = v - c;
  var rgb: vec3f;
  let hi = u32(h * 6.0) % 6u;
  switch(hi) {
    case 0u: { rgb = vec3f(c, x, 0.0); }
    case 1u: { rgb = vec3f(x, c, 0.0); }
    case 2u: { rgb = vec3f(0.0, c, x); }
    case 3u: { rgb = vec3f(0.0, x, c); }
    case 4u: { rgb = vec3f(x, 0.0, c); }
    default: { rgb = vec3f(c, 0.0, x); }
  }
  return rgb + vec3f(m);
}

fn kinColor(lineage24: u32) -> vec3f {
  let spread = lineage24 ^ (lineage24 >> 8u) ^ (lineage24 >> 16u);
  return hashToMutedRgb(spread * 2246822519u + lineage24 * 3266489917u);
}

fn envColorFromNorm(envNorm: f32, insetEnv: f32) -> vec3f {
  let e2 = envNorm * envNorm;
  return vec3f(
    0.006 + e2 * 0.028 + envNorm * 0.015,
    0.010 + envNorm * 0.09 + e2 * 0.08,
    0.018 + envNorm * 0.11 + e2 * 0.10
  ) * insetEnv;
}

fn applyNeuralTint(base: vec3f, neuralState: u32) -> vec3f {
  var c = base;
  if (neuralState == 1u) {
    c = min(c + vec3f(0.18, 0.12, 0.03), vec3f(1.0));
  }
  if (neuralState == 2u) {
    c = mix(c, c * vec3f(0.72, 0.78, 1.08), 0.42);
  }
  return c;
}

fn isMasked(index: u32) -> bool {
  return componentMask[index] != 0u;
}

fn maskedEdge4(index: u32) -> bool {
  if (!isMasked(index)) { return false; }
  let x = index % u.width;
  let y = index / u.width;
  if (x == 0u || x + 1u >= u.width || y == 0u || y + 1u >= u.height) {
    return true;
  }
  let up = index - u.width;
  let dn = index + u.width;
  let lf = index - 1u;
  let rt = index + 1u;
  return (!isMasked(up) || !isMasked(dn) || !isMasked(lf) || !isMasked(rt));
}

@vertex
fn vert(@builtin(vertex_index) vi: u32) -> VertexOutput {
  var positions = array<vec2f, 4>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0,  1.0),
  );
  var uvs = array<vec2f, 4>(
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 0.0),
  );
  var out: VertexOutput;
  out.pos = vec4f(positions[vi], 0.0, 1.0);
  out.uv = uvs[vi];
  return out;
}

@fragment
fn frag(@location(0) uv: vec2f) -> @location(0) vec4f {
  let viewUV = (uv - 0.5) / u.viewZoom + vec2f(u.viewX, u.viewY) / vec2f(f32(u.width), f32(u.height)) + 0.5;

  if (viewUV.x < 0.0 || viewUV.x >= 1.0 || viewUV.y < 0.0 || viewUV.y >= 1.0) {
    return vec4f(0.018, 0.022, 0.038, 1.0);
  }

  let x = u32(viewUV.x * f32(u.width));
  let y = u32(viewUV.y * f32(u.height));
  let index = y * u.width + x;

  let cellFx = fract(viewUV.x * f32(u.width));
  let cellFy = fract(viewUV.y * f32(u.height));
  let edgeDist = min(min(cellFx, 1.0 - cellFx), min(cellFy, 1.0 - cellFy));
  let insetEnv = mix(0.88, 1.0, smoothstep(0.0, 0.11, edgeDist));
  let insetCell = mix(0.72, 1.0, smoothstep(0.0, 0.19, edgeDist));

  let env = envEnergy[index];
  let cellType = getCellType(index);
  let cellEnergy = getCellEnergy(index);
  let neuralState = getNeuralState(index);
  let orgId = getOrgId(index);
  let lineage = cellState[index * 8u + 7u] & 0xFFFFFFu;

  let envNorm = clamp((env - 3.2) / 5.0, 0.0, 1.0);
  let occupied = cellType > 0u && orgId > 0u;

  var color: vec3f;

  if (u.viewMode == 0u) {
    color = envColorFromNorm(envNorm, insetEnv);
    if (occupied) {
      let eNorm = clamp(cellEnergy / 50.0, 0.22, 1.0);
      let base = kinColor(lineage);
      color = applyNeuralTint(base * eNorm, neuralState);
      color = max(color, vec3f(0.12, 0.09, 0.07));
      color *= insetCell;
    }
  } else if (u.viewMode == 1u) {
    let envNormWide = clamp((env - 2.0) / 6.5, 0.0, 1.0);
    let eBase = envColorFromNorm(envNormWide, insetEnv);
    if (occupied) {
      let eNorm = clamp(cellEnergy / 50.0, 0.22, 1.0);
      let cellC = applyNeuralTint(max(kinColor(lineage) * eNorm, vec3f(0.12, 0.09, 0.07)), neuralState) * insetCell;
      color = mix(eBase, cellC, 0.38);
    } else {
      color = eBase;
    }
  } else if (u.viewMode == 2u) {
    if (occupied) {
      let a = clamp(getMorphA(index) / 5.0, 0.0, 1.0);
      let b = clamp(getMorphB(index) / 5.0, 0.0, 1.0);
      color = vec3f(0.05 + a * 0.9, 0.06 + b * 0.88, 0.06 + (a + b) * 0.08);
      color = applyNeuralTint(color, neuralState);
      color *= insetCell;
    } else {
      color = envColorFromNorm(envNorm, insetEnv) * 0.52;
    }
  } else if (u.viewMode == 3u) {
    if (occupied) {
      let s = clamp(getStomach(index) / 14.0, 0.0, 1.0);
      color = mix(vec3f(0.14, 0.07, 0.24), vec3f(0.95, 0.82, 0.28), s);
      color = applyNeuralTint(color, neuralState);
      color *= insetCell;
    } else {
      color = envColorFromNorm(envNorm, insetEnv) * 0.48;
    }
  } else if (u.viewMode == 4u) {
    if (occupied) {
      let mk = getMarkerRgb(index);
      color = vec3f(mk.x * 0.92 + 0.05, mk.y * 0.92 + 0.04, mk.z * 0.92 + 0.06);
      color = applyNeuralTint(color, neuralState);
      color *= insetCell;
    } else {
      color = envColorFromNorm(envNorm, insetEnv) * 0.42;
    }
  } else {
    // Mode 5: energy grayscale + light lineage tint
    if (occupied) {
      let g = clamp(cellEnergy / 48.0, 0.1, 1.0);
      let tint = kinColor(lineage) * 0.24 * g;
      color = (vec3f(g * 0.93, g * 0.9, g * 0.86) + tint) * insetCell;
      color = applyNeuralTint(color, neuralState);
    } else {
      color = envColorFromNorm(envNorm, insetEnv) * 0.44;
    }
  }

  color = min(color * 1.04, vec3f(1.0));

  // Component highlight overlay (thin outline + soft fill).
  if (isMasked(index)) {
    let fill = vec3f(0.18, 0.40, 0.95);
    let edgeBoost = select(0.0, 1.0, maskedEdge4(index));
    // Sub-cell edge emphasis: stronger near pixel edges.
    let sub = 1.0 - smoothstep(0.03, 0.14, edgeDist);
    let outline = edgeBoost * (0.55 + 0.45 * sub);
    let alphaFill = 0.14;
    color = mix(color, min(vec3f(1.0), color + fill * alphaFill), 1.0);
    color = mix(color, min(vec3f(1.0), fill), outline);
  }
  return vec4f(color, 1.0);
}
