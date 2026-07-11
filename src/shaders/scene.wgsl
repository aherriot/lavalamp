const PHYSICS_BLOB_COUNT = 3u;
const TOTAL_BLOB_COUNT = 4u;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct SimParams {
  time: f32,
  dt: f32,
  resolution: vec2f,
  mouse: vec2f,
};

struct Blob {
  pos: vec2f,
  vel: vec2f,
};

@group(0) @binding(0) var<uniform> params: SimParams;

// ---- Compute stage: advances the two buoyant blobs' physics on the
// GPU. Runs once per frame, before the render pass reads the result. ----

@group(0) @binding(1) var<storage, read_write> blobsRW: array<Blob>;

@compute @workgroup_size(PHYSICS_BLOB_COUNT)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  var b = blobsRW[i];
  let fi = f32(i);

  // Same damped-spring buoyancy as the JS version from Step 7 -- only
  // where it runs has changed.
  let springStrength = 1.5;
  let damping = 0.8;
  let targetY = sin(params.time * (0.7 + fi * 0.3) + fi * 2.1) * 0.3;
  let ay = (targetY - b.pos.y) * springStrength - b.vel.y * damping;
  b.vel.y += ay * params.dt;
  b.pos.y += b.vel.y * params.dt;

  let ax = sin(params.time * (0.5 + fi * 0.2) + fi) * 0.1;
  b.vel.x += ax * params.dt;
  b.vel.x *= 0.98;
  b.pos.x += b.vel.x * params.dt;

  blobsRW[i] = b;
}

// ---- Render stage: unchanged conceptually from Step 7, just reads
// blob positions from the storage buffer instead of a uniform array. ----

@group(0) @binding(1) var<storage, read> blobsRO: array<Blob>;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0),
  );

  var out: VertexOutput;
  out.position = vec4f(pos[vertexIndex], 0.0, 1.0);
  out.uv = pos[vertexIndex] * 0.5 + 0.5;
  return out;
}

// Signed distance to a circle: negative inside, zero on the edge,
// positive outside. Any function with this shape -- negative inside,
// zero at the boundary, positive outside -- is a "signed distance field".
fn sdCircle(p: vec2f, radius: f32) -> f32 {
  return length(p) - radius;
}

// Smooth minimum: like min(a, b), but blends smoothly between the two
// instead of switching abruptly, with k controlling the blend radius.
// Applied to two SDFs, this is what makes two separate shapes visually
// merge into one wherever they get close -- the core "metaball" trick.
fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let aspect = params.resolution.x / params.resolution.y;

  // Center the coordinate space at (0,0) and correct for aspect ratio
  // so circles aren't stretched into ellipses on non-square canvases.
  var p = in.uv - vec2f(0.5, 0.5);
  p.x *= aspect;

  var radii = array<f32, TOTAL_BLOB_COUNT>(0.15, 0.12, 0.18, 0.15);

  let k = 0.15;
  var d = 1e5;
  for (var i = 0u; i < PHYSICS_BLOB_COUNT; i++) {
    var bp = blobsRO[i].pos;
    bp.x *= aspect;
    let bd = sdCircle(p - bp, radii[i]);
    d = smin(d, bd, k);
  }

  // The mouse-tracked blob isn't simulated state -- it's live input --
  // so it stays a plain uniform rather than living in the storage buffer.
  var mouseP = params.mouse * 0.5;
  mouseP.x *= aspect;
  let mouseD = sdCircle(p - mouseP, radii[TOTAL_BLOB_COUNT - 1u]);
  d = smin(d, mouseD, k);

  // Antialiasing via fwidth: it estimates how much `d` changes between
  // neighboring pixels, so the edge stays exactly ~1 pixel wide no
  // matter the screen resolution or how steeply d varies.
  let edge = fwidth(d);
  let coverage = 1.0 - smoothstep(-edge, edge, d);

  let background = vec3f(0.05, 0.05, 0.08);
  let blobColor = vec3f(1.0, 0.35 + 0.3 * sin(params.time), 0.2);

  let color = mix(background, blobColor, coverage);
  return vec4f(color, 1.0);
}
