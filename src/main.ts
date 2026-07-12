import "./style.css";
import { initWebGPU } from "./webgpu";
import sceneShader from "./shaders/scene.wgsl?raw";

const PHYSICS_BLOB_COUNT = 3;

async function main() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  const shaderModule = device.createShaderModule({ code: sceneShader });

  // layout: time: f32 (0), dt: f32 (4), resolution: vec2f (8),
  // cameraAzimuth: f32 (16), cameraElevation: f32 (20) -> 24 bytes.
  // No arrays here, so none of the uniform-array 16-byte-stride
  // padding from Step 7 applies.
  const uniformBuffer = device.createBuffer({
    size: 24,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Storage buffers don't have the uniform buffer's forced 16-byte
  // array stride -- each Blob (pos: vec2f, vel: vec2f) packs tightly
  // into 16 bytes, 4 floats, no padding required.
  const blobBufferSize =
    PHYSICS_BLOB_COUNT * 4 * Float32Array.BYTES_PER_ELEMENT;
  const blobBuffer = device.createBuffer({
    size: blobBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    blobBuffer,
    0,
    new Float32Array([
      -0.2,
      0.2,
      0,
      0, // blob 0: pos, vel
      0.15,
      -0.95,
      0,
      0, // blob 1: pos, vel
      0.0,
      -0.9,
      0,
      0, // blob 2: pos, vel
    ]),
  );

  // Orbit camera state: azimuth/elevation persist across drags, so
  // releasing the mouse mid-drag keeps auto-rotating smoothly from
  // wherever you left it rather than snapping back.
  const orbit = { azimuth: 0, elevation: 0.3, dragging: false };
  let lastPointer = { x: 0, y: 0 };

  const ORBIT_SENSITIVITY = 0.008;
  const ELEVATION_LIMIT = 1.4; // radians; keeps the camera short of the poles

  canvas.addEventListener("pointerdown", (event) => {
    orbit.dragging = true;
    lastPointer = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!orbit.dragging) {
      return;
    }
    const dx = event.clientX - lastPointer.x;
    const dy = event.clientY - lastPointer.y;
    lastPointer = { x: event.clientX, y: event.clientY };

    orbit.azimuth -= dx * ORBIT_SENSITIVITY;
    orbit.elevation = Math.max(
      -ELEVATION_LIMIT,
      Math.min(ELEVATION_LIMIT, orbit.elevation + dy * ORBIT_SENSITIVITY),
    );
  });
  const stopDragging = () => {
    orbit.dragging = false;
  };
  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);

  const computePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: shaderModule,
      entryPoint: "cs_main",
    },
  });

  const renderPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: {
      topology: "triangle-list",
    },
  });

  // Bloom: two more fullscreen passes reusing vs_main. Both read a
  // previous pass's output back as a texture, so unlike renderPipeline
  // (which renders straight to the canvas), this pass renders to an
  // offscreen texture that gets sampled afterward.
  const bloomExtractPipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: {
      module: shaderModule,
      entryPoint: "bloomExtract_fs",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const compositePipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: shaderModule, entryPoint: "vs_main" },
    fragment: {
      module: shaderModule,
      entryPoint: "composite_fs",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // Two separate bind groups over the *same* buffers: the compute
  // shader declares the storage buffer read_write, the fragment shader
  // declares it read-only, so each pipeline needs its own layout.
  const computeBindGroup = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: blobBuffer } },
    ],
  });

  // fs_main reads blob positions from the same storage buffer the
  // compute shader writes, reconnected as of Step 13.
  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: blobBuffer } },
    ],
  });

  // Offscreen textures the scene renders into, and the bright-pass
  // blur renders into, instead of straight to the canvas. Recreated
  // whenever the canvas resizes, since a texture's size is fixed at
  // creation time -- unlike the canvas itself, which webgpu.ts
  // reconfigures in place on resize.
  let sceneTexture: GPUTexture | null = null;
  let bloomTexture: GPUTexture | null = null;
  let bloomExtractBindGroup: GPUBindGroup | null = null;
  let compositeBindGroup: GPUBindGroup | null = null;
  let offscreenWidth = 0;
  let offscreenHeight = 0;

  function ensureOffscreenTargets() {
    if (canvas.width === offscreenWidth && canvas.height === offscreenHeight) {
      return;
    }
    offscreenWidth = canvas.width;
    offscreenHeight = canvas.height;

    sceneTexture?.destroy();
    bloomTexture?.destroy();

    sceneTexture = device.createTexture({
      size: [offscreenWidth, offscreenHeight],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    bloomTexture = device.createTexture({
      size: [offscreenWidth, offscreenHeight],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    bloomExtractBindGroup = device.createBindGroup({
      layout: bloomExtractPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: sampler },
        { binding: 3, resource: sceneTexture.createView() },
      ],
    });

    compositeBindGroup = device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 2, resource: sampler },
        { binding: 3, resource: sceneTexture.createView() },
        { binding: 4, resource: bloomTexture.createView() },
      ],
    });
  }

  const uniformData = new Float32Array(6);
  let lastTimeMs = 0;

  function frame(timeMs: number) {
    ensureOffscreenTargets();

    const t = timeMs * 0.001;
    const dt = Math.min((timeMs - lastTimeMs) * 0.001, 0.05);
    lastTimeMs = timeMs;

    if (!orbit.dragging) {
      orbit.azimuth += 0.15 * dt;
    }

    uniformData[0] = t;
    uniformData[1] = dt;
    uniformData[2] = canvas.width;
    uniformData[3] = canvas.height;
    uniformData[4] = orbit.azimuth;
    uniformData[5] = orbit.elevation;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(1);
    computePass.end();

    // Pass 1: render the actual scene into an offscreen texture
    // instead of the canvas.
    const scenePass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: sceneTexture!.createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    scenePass.setPipeline(renderPipeline);
    scenePass.setBindGroup(0, renderBindGroup);
    scenePass.draw(3);
    scenePass.end();

    // Pass 2: extract + blur the bright parts of that scene texture
    // into a second offscreen texture.
    const bloomPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: bloomTexture!.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    bloomPass.setPipeline(bloomExtractPipeline);
    bloomPass.setBindGroup(0, bloomExtractBindGroup!);
    bloomPass.draw(3);
    bloomPass.end();

    // Pass 3: composite scene + bloom together onto the actual canvas.
    const compositePass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    compositePass.setPipeline(compositePipeline);
    compositePass.setBindGroup(0, compositeBindGroup!);
    compositePass.draw(3);
    compositePass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  document.body.innerHTML = `<pre style="color:#f66;padding:2rem;font:14px monospace">${String(err)}</pre>`;
});
