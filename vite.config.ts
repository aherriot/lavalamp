import { createServer as createHttpServer } from "http";
import { defineConfig, type Plugin } from "vite";
import mkcert from "vite-plugin-mkcert";

const HTTPS_PORT = 5173;
const HTTP_PORT = 5172;

// Vite's own dev server only speaks one protocol at a time, so plain
// http requests just fail to connect once https is enabled. This spins
// up a second, minimal server on an adjacent port that does nothing but
// 301-redirect every request over to the https origin.
function httpToHttpsRedirect(): Plugin {
  return {
    name: "http-to-https-redirect",
    configureServer() {
      createHttpServer((req, res) => {
        const host = (req.headers.host ?? "localhost").split(":")[0];
        res.writeHead(301, {
          Location: `https://${host}:${HTTPS_PORT}${req.url ?? "/"}`,
        });
        res.end();
      }).listen(HTTP_PORT);
    },
  };
}

export default defineConfig({
  plugins: [mkcert(), httpToHttpsRedirect()],
  server: { port: HTTPS_PORT },
});
