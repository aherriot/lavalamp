struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct Uniforms {
  time: f32,
  mouse: vec2f,
  resolution: vec2f,
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let aspect = uniforms.resolution.x / uniforms.resolution.y;

  // Center the coordinate space at (0,0) and correct for aspect ratio
  // so the circle isn't stretched into an ellipse on non-square canvases.
  var p = in.uv - vec2f(0.5, 0.5);
  p.x *= aspect;

  var mouseP = uniforms.mouse * 0.5;
  mouseP.x *= aspect;

  let d = sdCircle(p - mouseP, 0.2);

  // Antialiasing via fwidth: it estimates how much `d` changes between
  // neighboring pixels, so the edge stays exactly ~1 pixel wide no
  // matter the screen resolution or how steeply d varies.
  let edge = fwidth(d);
  let coverage = 1.0 - smoothstep(-edge, edge, d);

  let background = vec3f(0.05, 0.05, 0.08);
  let circleColor = vec3f(1.0, 0.35 + 0.3 * sin(uniforms.time), 0.2);

  let color = mix(background, circleColor, coverage);
  return vec4f(color, 1.0);
}
