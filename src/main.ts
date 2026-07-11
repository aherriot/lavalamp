import "./style.css";
import { initWebGPU } from "./webgpu";
import sceneShader from "./shaders/scene.wgsl?raw";

const BLOB_COUNT = 3;

interface Blob {
  pos: [number, number];
  vel: [number, number];
}

async function main() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  const shaderModule = device.createShaderModule({ code: sceneShader });

  // layout: time: f32 (offset 0), resolution: vec2f (offset 8),
  // blobPositions: array<vec2f, 3> (offset 16, 16 bytes per element
  // instead of 8 -- WGSL forces uniform-buffer array strides to be a
  // multiple of 16 bytes, so each vec2f entry gets 8 bytes of padding).
  // Total: 16 + 3 * 16 = 64 bytes.
  const uniformBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const mouseNDC = { x: 0, y: 0 };
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    mouseNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouseNDC.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  });

  // Two blobs driven by simple damped-spring "buoyancy": each is pulled
  // toward a slowly oscillating target height, with damping so it
  // settles into a smooth bob instead of oscillating forever.
  const buoyantBlobs: Blob[] = [
    { pos: [-0.2, 0.1], vel: [0, 0] },
    { pos: [0.15, -0.15], vel: [0, 0] },
  ];

  function stepPhysics(t: number, dt: number) {
    for (let i = 0; i < buoyantBlobs.length; i++) {
      const b = buoyantBlobs[i];
      const springStrength = 1.5;
      const damping = 0.8;

      const targetY = Math.sin(t * (0.7 + i * 0.3) + i * 2.1) * 0.3;
      const ay = (targetY - b.pos[1]) * springStrength - b.vel[1] * damping;
      b.vel[1] += ay * dt;
      b.pos[1] += b.vel[1] * dt;

      const ax = Math.sin(t * (0.5 + i * 0.2) + i) * 0.1;
      b.vel[0] += ax * dt;
      b.vel[0] *= 0.98; // friction, otherwise horizontal drift accumulates forever
      b.pos[0] += b.vel[0] * dt;
    }
  }

  const pipeline = device.createRenderPipeline({
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

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  const uniformData = new Float32Array(16);
  let lastTimeMs = 0;

  function frame(timeMs: number) {
    const t = timeMs * 0.001;
    const dt = Math.min((timeMs - lastTimeMs) * 0.001, 0.05);
    lastTimeMs = timeMs;

    stepPhysics(t, dt);

    // World space here matches the shader's `p`, which ranges roughly
    // -0.5..0.5 (uv - 0.5), so the mouse blob is scaled down to match.
    const blobPositions: [number, number][] = [
      buoyantBlobs[0].pos,
      buoyantBlobs[1].pos,
      [mouseNDC.x * 0.5, mouseNDC.y * 0.5],
    ];

    uniformData[0] = t;
    uniformData[2] = canvas.width;
    uniformData[3] = canvas.height;
    for (let i = 0; i < BLOB_COUNT; i++) {
      const base = 4 + i * 4; // 4 floats (16 bytes) per array element
      uniformData[base] = blobPositions[i][0];
      uniformData[base + 1] = blobPositions[i][1];
    }
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  document.body.innerHTML = `<pre style="color:#f66;padding:2rem;font:14px monospace">${String(err)}</pre>`;
});
