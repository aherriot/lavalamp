import "./style.css";
import { initWebGPU } from "./webgpu";
import sceneShader from "./shaders/scene.wgsl?raw";

async function main() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  const shaderModule = device.createShaderModule({ code: sceneShader });

  // layout: time: f32 (offset 0), mouse: vec2f (offset 8), resolution:
  // vec2f (offset 16) -> 24 bytes total. Each vec2f must start on an
  // 8-byte boundary per WGSL's uniform alignment rules.
  const uniformBuffer = device.createBuffer({
    size: 24,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const mouseNDC = { x: 0, y: 0 };
  canvas.addEventListener("pointermove", (event) => {
    const rect = canvas.getBoundingClientRect();
    mouseNDC.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouseNDC.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  });

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

  function frame(timeMs: number) {
    const t = timeMs * 0.001;
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      new Float32Array([
        t,
        0,
        mouseNDC.x,
        mouseNDC.y,
        canvas.width,
        canvas.height,
      ]),
    );

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
