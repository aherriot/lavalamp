struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

const BLOB_COUNT = 3u;

struct Uniforms {
  time: f32,
  resolution: vec2f,
  // NOTE: arrays inside a uniform buffer must have an element stride
  // that's a multiple of 16 bytes, even though a bare vec2f is only
  // 8 bytes. WGSL pads each array entry to 16 bytes here -- the JS
  // side has to match that padding by hand when writing the buffer.
  blobPositions: array<vec2f, BLOB_COUNT>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

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
  let aspect = uniforms.resolution.x / uniforms.resolution.y;

  // Center the coordinate space at (0,0) and correct for aspect ratio
  // so circles aren't stretched into ellipses on non-square canvases.
  var p = in.uv - vec2f(0.5, 0.5);
  p.x *= aspect;

  // Blob positions and radii now come from JS-side physics instead of
  // sin/cos formulas baked into the shader.
  var radii = array<f32, BLOB_COUNT>(0.15, 0.12, 0.18);

  let k = 0.15;
  var d = 1e5;
  for (var i = 0u; i < BLOB_COUNT; i++) {
    var bp = uniforms.blobPositions[i];
    bp.x *= aspect;
    let bd = sdCircle(p - bp, radii[i]);
    d = smin(d, bd, k);
  }

  // Antialiasing via fwidth: it estimates how much `d` changes between
  // neighboring pixels, so the edge stays exactly ~1 pixel wide no
  // matter the screen resolution or how steeply d varies.
  let edge = fwidth(d);
  let coverage = 1.0 - smoothstep(-edge, edge, d);

  let background = vec3f(0.05, 0.05, 0.08);
  let blobColor = vec3f(1.0, 0.35 + 0.3 * sin(uniforms.time), 0.2);

  let color = mix(background, blobColor, coverage);
  return vec4f(color, 1.0);
}
