import "./style.css";
import { initWebGPU } from "./webgpu";
import sceneShader from "./shaders/scene.wgsl?raw";

const PHYSICS_BLOB_COUNT = 3;

async function main() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  const shaderModule = device.createShaderModule({ code: sceneShader });

  // layout: time: f32 (0), dt: f32 (4), resolution: vec2f (8),
  // mouse: vec2f (16) -> 24 bytes. No arrays here, so none of the
  // uniform-array 16-byte-stride padding from Step 7 applies.
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
      0.1,
      0,
      0, // blob 0: pos, vel
      0.15,
      -0.15,
      0,
      0, // blob 1: pos, vel
      0.0,
      -0.5,
      0,
      0, // blob 2: pos, vel
    ]),
  );

  const mouseNDC = { x: 0, y: 0 };
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    mouseNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouseNDC.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  });

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

  const renderBindGroup = device.createBindGroup({
    layout: renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: blobBuffer } },
    ],
  });

  const uniformData = new Float32Array(6);
  let lastTimeMs = 0;

  function frame(timeMs: number) {
    const t = timeMs * 0.001;
    const dt = Math.min((timeMs - lastTimeMs) * 0.001, 0.05);
    lastTimeMs = timeMs;

    uniformData[0] = t;
    uniformData[1] = dt;
    uniformData[2] = canvas.width;
    uniformData[3] = canvas.height;
    uniformData[4] = mouseNDC.x;
    uniformData[5] = mouseNDC.y;
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(1);
    computePass.end();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(3);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  document.body.innerHTML = `<pre style="color:#f66;padding:2rem;font:14px monospace">${String(err)}</pre>`;
});
