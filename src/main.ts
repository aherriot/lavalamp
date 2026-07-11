import './style.css';
import { initWebGPU } from './webgpu';

async function main() {
  const canvas = document.getElementById('gpu-canvas') as HTMLCanvasElement;
  const { device, context } = await initWebGPU(canvas);

  function frame(timeMs: number) {
    const t = timeMs * 0.001;

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: {
            r: 0.5 + 0.5 * Math.sin(t),
            g: 0.1,
            b: 0.5 + 0.5 * Math.cos(t),
            a: 1,
          },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
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
