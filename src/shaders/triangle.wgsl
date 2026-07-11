struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
};

struct Uniforms {
  time: f32,
  mouse: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(
  @location(0) position: vec2f,
  @location(1) color: vec3f,
) -> VertexOutput {
  let angle = uniforms.time * uniforms.mouse.x * 1.0;
  let c = cos(angle);
  let s = sin(angle);
  let rotated = vec2f(
    position.x * c - position.y * s,
    position.x * s + position.y * c,
  );

  var out: VertexOutput;
  out.position = vec4f(rotated, 0.0, 1.0);
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  return vec4f(in.color, 1.0);
}
