import './style.css';
import { initWebGPU } from './webgpu';
import triangleShader from './shaders/triangle.wgsl?raw';

async function main() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  const { device, context, format } = await initWebGPU(canvas);

  const shaderModule = device.createShaderModule({ code: triangleShader });

  const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{ format }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  function frame() {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });

    pass.setPipeline(pipeline);
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
