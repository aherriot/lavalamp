import "./style.css";
import { initWebGPU } from "./webgpu";
import triangleShader from "./shaders/triangle.wgsl?raw";

async function main() {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  const shaderModule = device.createShaderModule({ code: triangleShader });

  // Interleaved per-vertex data: [x, y, r, g, b] x 3 vertices.
  const vertexData = new Float32Array([
    0.0, 0.5, 1.0, 0.2, 0.2, -0.5, -0.5, 0.2, 1.0, 0.2, 0.5, -0.5, 0.2, 0.4,
    1.0,
  ]);
  const vertexBuffer = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vertexBuffer, 0, vertexData);

  // Uint16Array of 3 indices is 6 bytes; GPU buffer writes must be
  // 4-byte aligned, so we pad with one unused index.
  const indexData = new Uint16Array([0, 1, 2, 0]);
  const indexBuffer = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indexBuffer, 0, indexData);

  // layout: time: f32 (offset 0), mouse: vec2f (offset 8, per WGSL's
  // 8-byte alignment for vec2f) -> 16 bytes total.
  const uniformBuffer = device.createBuffer({
    size: 16,
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
      buffers: [
        {
          arrayStride: 5 * Float32Array.BYTES_PER_ELEMENT,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            {
              shaderLocation: 1,
              offset: 2 * Float32Array.BYTES_PER_ELEMENT,
              format: "float32x3",
            },
          ],
        },
      ],
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
      new Float32Array([t, 0, mouseNDC.x, mouseNDC.y]),
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
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, "uint16");
    pass.drawIndexed(3);
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
