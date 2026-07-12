import "./style.css";
import { initWebGPU } from "./webgpu";
import sceneShaderTemplate from "./shaders/scene.wgsl?raw";
import { Pane } from "tweakpane";

const MIN_BLOB_COUNT = 1;
const MAX_BLOB_COUNT = 24;
const DEFAULT_BLOB_COUNT = 12;

interface Params {
  blobCount: number;
  animSpeed: number;
  autoRotateSpeed: number;
  lightX: number;
  lightY: number;
  lightZ: number;
  bloomThreshold: number;
  bloomIntensity: number;
  maxSteps: number;
  hitEpsilon: number;
  coolR: number;
  coolG: number;
  coolB: number;
  warmR: number;
  warmG: number;
  warmB: number;
  hotR: number;
  hotG: number;
  hotB: number;
}

const params: Params = {
  blobCount: DEFAULT_BLOB_COUNT,
  animSpeed: 3.0,
  autoRotateSpeed: 0.15,
  lightX: 2.0,
  lightY: 3.0,
  lightZ: 2.0,
  bloomThreshold: 0.6,
  bloomIntensity: 1.2,
  maxSteps: 100,
  hitEpsilon: 0.001,
  // Classic red-wax lava lamp palette: deep maroon at rest, brightening
  // through red-orange to a warm amber glow when "hot".
  coolR: 0.5,
  coolG: 0.03,
  coolB: 0.03,
  warmR: 0.85,
  warmG: 0.15,
  warmB: 0.05,
  hotR: 1.0,
  hotG: 0.5,
  hotB: 0.15,
};

async function main() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  // Builds the WGSL source with the blob count baked in. WGSL array
  // sizes and @workgroup_size must be known at shader-compile time, so
  // a GUI-driven blob-count slider can't just be another uniform --
  // changing it means substituting a new value into the source text
  // and recreating the shader module (and everything downstream of it)
  // from scratch.
  function buildShaderSource(blobCount: number): string {
    return sceneShaderTemplate.replace(/__BLOB_COUNT__/g, String(blobCount));
  }

  // Everything that depends on the blob count: the shader module
  // itself, all four pipelines built from it, and the storage buffer
  // sized to match. Bundled together since they all need rebuilding
  // in lockstep whenever the count changes.
  function createBlobPipelineSet(blobCount: number) {
    const shaderModule = device.createShaderModule({
      code: buildShaderSource(blobCount),
    });

    const computePipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shaderModule, entryPoint: "cs_main" },
    });

    const renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: shaderModule, entryPoint: "vs_main" },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });

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

    // Storage buffers pack tightly (no forced 16-byte array stride
    // like uniform buffers have): 4 floats per blob.
    const blobBuffer = device.createBuffer({
      size: blobCount * 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const initialBlobs = new Float32Array(blobCount * 4);
    for (let i = 0; i < blobCount; i++) {
      const base = i * 4;
      const spreadT = blobCount > 1 ? i / (blobCount - 1) : 0.5;
      // Evenly spread from top to bottom (matching the widened vertical
      // range physics now uses), plus a little jitter so it doesn't
      // read as a perfectly even staircase before the sim takes over.
      const jitter = (((i * 37) % 5) - 2) * 0.05;
      // x used to be a plain linear function of i too -- since y also
      // increases monotonically with i, every blob's (x, y) moved
      // together, reading as a straight diagonal line rather than a
      // 2D spread. sin() of a golden-angle-ish step scatters x
      // non-monotonically across index instead, so consecutive blobs
      // land at unrelated horizontal positions.
      initialBlobs[base] = Math.sin(i * 2.4) * 0.5; // pos.x, scattered
      initialBlobs[base + 1] = -1.1 + spreadT * 2.2 + jitter; // pos.y, spread top-to-bottom
      initialBlobs[base + 2] = 0; // vel.x
      initialBlobs[base + 3] = 0; // vel.y
    }
    device.queue.writeBuffer(blobBuffer, 0, initialBlobs);

    return {
      shaderModule,
      computePipeline,
      renderPipeline,
      bloomExtractPipeline,
      compositePipeline,
      blobBuffer,
    };
  }

  let gpu = createBlobPipelineSet(params.blobCount);

  // layout: time(0) dt(4) resolution(8,vec2f) cameraAzimuth(16)
  // cameraElevation(20) maxSteps(24) hitEpsilon(28) lightPos(32,vec4f)
  // coolColor(48,vec4f) warmColor(64,vec4f) hotColor(80,vec4f) -> 96
  // bytes. lightPos.w and coolColor.w double up as bloom
  // threshold/intensity -- see the SimParams struct comment in
  // scene.wgsl for why.
  const uniformBuffer = device.createBuffer({
    size: 96,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });

  // Two separate bind groups over the *same* buffers: the compute
  // shader declares the storage buffer read_write, the fragment shader
  // declares it read-only, so each pipeline needs its own layout.
  let computeBindGroup = device.createBindGroup({
    layout: gpu.computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gpu.blobBuffer } },
    ],
  });
  let renderBindGroup = device.createBindGroup({
    layout: gpu.renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: gpu.blobBuffer } },
    ],
  });

  function rebuildBlobCount(newCount: number) {
    gpu.blobBuffer.destroy();
    gpu = createBlobPipelineSet(newCount);
    computeBindGroup = device.createBindGroup({
      layout: gpu.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gpu.blobBuffer } },
      ],
    });
    renderBindGroup = device.createBindGroup({
      layout: gpu.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: gpu.blobBuffer } },
      ],
    });
    // Forces ensureOffscreenTargets() to recreate the bloom/composite
    // bind groups too, since they reference gpu.bloomExtractPipeline /
    // gpu.compositePipeline layouts, which just changed.
    offscreenWidth = 0;
    offscreenHeight = 0;
  }

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

  // Offscreen textures the scene renders into, and the bright-pass
  // blur renders into, instead of straight to the canvas. Recreated
  // whenever the canvas resizes (texture size is fixed at creation
  // time, unlike the canvas) or the blob count changes (pipelines --
  // and therefore bind-group layouts -- get rebuilt from scratch).
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
      layout: gpu.bloomExtractPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: sampler },
        { binding: 3, resource: sceneTexture.createView() },
      ],
    });

    compositeBindGroup = device.createBindGroup({
      layout: gpu.compositePipeline.getBindGroupLayout(0),
      entries: [
        // composite_fs reads params.coolColor.w for bloom intensity,
        // so its auto-inferred layout now expects binding 0 too --
        // easy to miss since this wasn't required before that change.
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: sampler },
        { binding: 3, resource: sceneTexture.createView() },
        { binding: 4, resource: bloomTexture.createView() },
      ],
    });
  }

  // ---- Tweakpane GUI ----
  const pane = new Pane({ title: "Lava Lamp Controls" });

  const simFolder = pane.addFolder({ title: "Simulation" });
  simFolder
    .addBinding(params, "blobCount", {
      min: MIN_BLOB_COUNT,
      max: MAX_BLOB_COUNT,
      step: 1,
    })
    .on("change", (ev) => {
      // Rebuilding recompiles four pipelines from scratch -- only do
      // it once the slider drag actually finishes (ev.last), not on
      // every intermediate value while dragging.
      if (ev.last) {
        rebuildBlobCount(Math.round(ev.value as number));
      }
    });
  simFolder.addBinding(params, "animSpeed", { min: 0, max: 6, step: 0.05 });
  simFolder.addBinding(params, "autoRotateSpeed", {
    min: 0,
    max: 1,
    step: 0.01,
  });

  const lightFolder = pane.addFolder({ title: "Light position" });
  lightFolder.addBinding(params, "lightX", { min: -5, max: 5, step: 0.1 });
  lightFolder.addBinding(params, "lightY", { min: -5, max: 5, step: 0.1 });
  lightFolder.addBinding(params, "lightZ", { min: -5, max: 5, step: 0.1 });

  const colorFolder = pane.addFolder({ title: "Lava colors", expanded: false });
  colorFolder.addBinding(params, "coolR", { min: 0, max: 1, step: 0.01, label: "cool r" });
  colorFolder.addBinding(params, "coolG", { min: 0, max: 1, step: 0.01, label: "cool g" });
  colorFolder.addBinding(params, "coolB", { min: 0, max: 1, step: 0.01, label: "cool b" });
  colorFolder.addBinding(params, "warmR", { min: 0, max: 1, step: 0.01, label: "warm r" });
  colorFolder.addBinding(params, "warmG", { min: 0, max: 1, step: 0.01, label: "warm g" });
  colorFolder.addBinding(params, "warmB", { min: 0, max: 1, step: 0.01, label: "warm b" });
  colorFolder.addBinding(params, "hotR", { min: 0, max: 1, step: 0.01, label: "hot r" });
  colorFolder.addBinding(params, "hotG", { min: 0, max: 1, step: 0.01, label: "hot g" });
  colorFolder.addBinding(params, "hotB", { min: 0, max: 1, step: 0.01, label: "hot b" });

  const glowFolder = pane.addFolder({ title: "Glow & perf", expanded: false });
  glowFolder.addBinding(params, "bloomThreshold", { min: 0, max: 1.5, step: 0.01 });
  glowFolder.addBinding(params, "bloomIntensity", { min: 0, max: 3, step: 0.05 });
  glowFolder.addBinding(params, "maxSteps", {
    min: 8,
    max: 150,
    step: 1,
    label: "raymarch steps",
  });
  glowFolder.addBinding(params, "hitEpsilon", {
    min: 0.0001,
    max: 0.02,
    step: 0.0001,
    label: "hit epsilon",
  });

  const uniformData = new Float32Array(24);
  let lastTimeMs = 0;
  let simulatedTime = 0;

  function frame(timeMs: number) {
    ensureOffscreenTargets();

    const rawDt = Math.min((timeMs - lastTimeMs) * 0.001, 0.05);
    lastTimeMs = timeMs;
    const dt = rawDt * params.animSpeed;
    simulatedTime += dt;

    if (!orbit.dragging) {
      orbit.azimuth += params.autoRotateSpeed * rawDt;
    }

    uniformData[0] = simulatedTime;
    uniformData[1] = dt;
    uniformData[2] = canvas.width;
    uniformData[3] = canvas.height;
    uniformData[4] = orbit.azimuth;
    uniformData[5] = orbit.elevation;
    uniformData[6] = params.maxSteps;
    uniformData[7] = params.hitEpsilon;
    uniformData[8] = params.lightX;
    uniformData[9] = params.lightY;
    uniformData[10] = params.lightZ;
    uniformData[11] = params.bloomThreshold;
    uniformData[12] = params.coolR;
    uniformData[13] = params.coolG;
    uniformData[14] = params.coolB;
    uniformData[15] = params.bloomIntensity;
    uniformData[16] = params.warmR;
    uniformData[17] = params.warmG;
    uniformData[18] = params.warmB;
    uniformData[19] = 0;
    uniformData[20] = params.hotR;
    uniformData[21] = params.hotG;
    uniformData[22] = params.hotB;
    uniformData[23] = 0;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(gpu.computePipeline);
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
    scenePass.setPipeline(gpu.renderPipeline);
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
    bloomPass.setPipeline(gpu.bloomExtractPipeline);
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
    compositePass.setPipeline(gpu.compositePipeline);
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
