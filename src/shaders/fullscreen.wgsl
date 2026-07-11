struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct Uniforms {
  time: f32,
  mouse: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  // A single triangle, deliberately oversized so it covers the whole
  // screen after clipping. No vertex buffer needed: three positions,
  // indexed straight from vertex_index.
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let uv = in.uv;

  // A slow color gradient across the screen, purely a function of
  // this pixel's uv and the current time.
  let base = vec3f(uv.x, uv.y, 0.5 + 0.5 * sin(uniforms.time));

  // A soft glow that follows the mouse -- a preview of the distance-field
  // techniques Step 5 introduces properly.
  let mouseUV = uniforms.mouse * 0.5 + 0.5;
  let d = distance(uv, mouseUV);
  let glow = smoothstep(0.2, 0.0, d);

  return vec4f(base + glow, 1.0);
}
