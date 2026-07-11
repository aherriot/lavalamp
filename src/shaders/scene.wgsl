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

// A pseudo-random hash: not truly random, but scattered enough that
// nearby inputs give unrelated outputs -- the building block every
// noise function starts from.
fn hash21(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// Value noise: hash the four corners of the grid cell containing p,
// then smoothly interpolate between them. Unlike hash21 alone (which
// jumps randomly pixel to pixel), this varies gradually -- the
// "smooth randomness" that makes motion look organic instead of jittery.
fn noise2D(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  let u = f * f * (3.0 - 2.0 * f); // ease the interpolation, avoids visible grid seams
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

@compute @workgroup_size(PHYSICS_BLOB_COUNT)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  var b = blobsRW[i];
  let fi = f32(i);

  // Heat field: slow-evolving noise sampled purely in time (each blob
  // offset so they don't move in lockstep). Deliberately *not* a
  // function of the blob's own position -- coupling heat to pos.x
  // creates a feedback loop (move -> heat changes -> accelerate ->
  // move faster) that reads as jittery instead of calm.
  let heat = noise2D(vec2f(fi * 5.0, params.time * 0.08)) * 2.0 - 1.0;

  // A gentle spring pulling back toward the vertical center. Unlike a
  // constant gravity term, this is proportional to displacement -- the
  // further a blob drifts from the middle, the harder it's pulled
  // back -- so blobs settle into hovering near center instead of
  // drifting to a wall and waiting to bounce off it.
  let restoreStrength = 0.6;
  let heatStrength = 0.5;
  var ay = (0.0 - b.pos.y) * restoreStrength + heat * heatStrength;
  var ax = 0.0;

  // Mild repulsion so blobs don't sit directly on top of each other.
  for (var j = 0u; j < PHYSICS_BLOB_COUNT; j++) {
    if (j == i) {
      continue;
    }
    let other = blobsRW[j].pos;
    let delta = b.pos - other;
    let dist = max(length(delta), 0.001);
    let minDist = 0.22;
    if (dist < minDist) {
      let push = (minDist - dist) * 2.0;
      ax += (delta.x / dist) * push;
      ay += (delta.y / dist) * push;
    }
  }

  b.vel.x += ax * params.dt;
  b.vel.y += ay * params.dt;

  // Exponential velocity decay (viscous drag) instead of a subtractive
  // acceleration term: this is what actually makes the motion feel
  // slow and syrupy rather than springy. exp(-rate * dt) decays the
  // same amount per second regardless of frame rate.
  b.vel *= exp(-1.2 * params.dt);

  b.pos += b.vel * params.dt;

  // Keep blobs inside a rough container, losing a bit of energy on bounce.
  let boundX = 0.42;
  let boundY = 0.42;
  if (abs(b.pos.x) > boundX) {
    b.pos.x = clamp(b.pos.x, -boundX, boundX);
    b.vel.x *= -0.4;
  }
  if (abs(b.pos.y) > boundY) {
    b.pos.y = clamp(b.pos.y, -boundY, boundY);
    b.vel.y *= -0.4;
  }

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
