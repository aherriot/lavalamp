# Lava Lamp in WebGPU — Lesson Plan

Learning computer graphics from first principles by building a fully 3D,
raymarched lava lamp in WebGPU + TypeScript, one small step at a time.

Stack: TypeScript + Vite. Final look: full 3D raymarched metaballs inside a
glass container, with lighting, refraction, and glow.

---

## Phase 0 — Setup

- [x] **0. Hello GPU** — Vite+TS scaffold, get a WebGPU adapter/device,
  configure the canvas, clear it to a color every frame.
  *Teaches:* adapter → device → queue model, canvas/swapchain configuration,
  command encoders & render passes — the skeleton every future step reuses.

## Phase 1 — Rasterization fundamentals

- [x] **1. First triangle** — hardcoded vertices inside WGSL itself, no
  buffers yet.
  *Teaches:* WGSL vertex/fragment shaders, pipelines, NDC space, what
  "rasterization" actually does.

- [x] **2. Vertex buffers** — same triangle, now driven by CPU-side vertex
  data (+ index buffer).
  *Teaches:* GPU buffers, vertex layouts, indexed drawing.

- [x] **3. Uniforms & animation** — rotate/move the shape every frame via a
  uniform buffer.
  *Teaches:* uniform buffers, bind groups/layouts, the animation loop, time
  as GPU input.

- [x] **4. The fullscreen quad trick** — render a single triangle that
  covers the screen, treat the fragment shader as a per-pixel program.
  *Teaches:* this is the technique the entire rest of the project is built
  on — once you have this, "rendering" becomes "writing a function of pixel
  coordinates."

## Phase 2 — SDFs & metaballs (2D warm-up)

- [x] **5. A circle from math** — draw a circle using a signed distance
  function inside the fragment shader.
  *Teaches:* what an SDF is, `smoothstep` antialiasing without any geometry
  at all.

- [x] **6. Blob merging** — several circles blended with a smooth-min
  function = classic 2D metaballs.
  *Teaches:* `smin`, blending functions — the exact trick that makes lava
  lamp blobs melt into each other later.

- [x] **7. Simple buoyancy** — blobs drift up/down with sine-wave wobble,
  positions passed in via a uniform array.
  *Teaches:* passing arrays to shaders, basic physics integration
  (position/velocity).
  *Side quest:* swap in a couple of different easing/noise functions to see
  how "organic" motion is faked cheaply.

## Phase 3 — Compute shaders

- [x] **8. Physics moves to the GPU** — blob positions/velocities live in a
  storage buffer, updated by a compute shader instead of JS.
  *Teaches:* compute pipelines, workgroups, storage buffers, the compute →
  render handoff (GPU talking to itself).

- [ ] **9. Heat-driven motion** — a simulated noise field drives buoyancy,
  blobs mildly repel each other.
  *Teaches:* noise functions in WGSL (value/Perlin-ish), lightweight
  N-body-style interaction.

## Phase 4 — Into 3D

- [ ] **10. Rays instead of matrices** — generate a camera ray per pixel
  from position/look-direction/FOV.
  *Teaches:* how raymarched renderers do "projection" without a traditional
  vertex pipeline.

- [ ] **11. Raymarching one sphere** — sphere-trace an SDF sphere, shade it
  with one light.
  *Teaches:* the raymarching algorithm itself, computing normals from the
  SDF gradient, basic Lambertian shading.

- [ ] **12. 3D metaballs** — multiple spheres + 3D smooth-min = actual lava
  blobs.
  *Teaches:* extending the Phase 2 blending trick into 3D scene
  composition.

- [ ] **13. Simulation meets renderer** — the compute-shader physics from
  Phase 3 now drives real 3D blob positions.
  *Teaches:* wiring a full simulate → raymarch pipeline, the core loop of
  the finished demo.

## Phase 5 — Look & feel

- [ ] **14. Real lighting** — ambient + diffuse + specular (Blinn-Phong),
  plus a rim light for that glowing look.
  *Teaches:* standard lighting model, material parameters.

- [ ] **15. The glass lamp** — add a capsule/cylinder SDF container, boolean
  ops (union/subtract), fake refraction & fresnel at the glass surface.
  *Teaches:* SDF booleans, cheap-but-convincing refraction/reflection
  tricks.

- [ ] **16. Color & glow** — height/heat-driven lava gradient, bloom via a
  render-to-texture + blur + additive-composite pass.
  *Teaches:* multi-pass rendering, the simplest real post-processing
  pipeline.

- [ ] **17. Orbit camera** — mouse-drag camera control.
  *Teaches:* basic interaction handling, spherical/quaternion camera math.

- [ ] **18. Polish & tuning** — tweak raymarch step count/epsilon for perf,
  add a live GUI (tweakpane) for blob count, colors, speed, light position.
  *Teaches:* the actual perf/quality tradeoffs raymarching lives and dies
  by, plus a nicety for demoing.

---

That's the full arc: rasterization → SDFs → compute → raymarching →
shading/post-fx, ending in a tunable, demoable 3D lava lamp.
