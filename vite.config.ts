import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// Dev only: report early page errors to the `tauri dev` terminal, before the app's modules load.
const devProbe = (): Plugin => ({
  name: 'dev-probe',
  apply: 'serve',
  transformIndexHtml: html =>
    html.replace(
      '<head>',
      `<head>
<script>
(function () {
  var send = function (m) { try { window.__TAURI_INTERNALS__ && window.__TAURI_INTERNALS__.invoke('js_log', { msg: String(m) }); } catch (e) {} };
  window.addEventListener('error', function (e) { send((e.message || 'error') + ' @ ' + (e.filename || (e.target && (e.target.src || e.target.href)) || '') + ':' + (e.lineno || '')); }, true);
  window.addEventListener('unhandledrejection', function (e) { send('unhandled: ' + (e.reason && (e.reason.stack || e.reason))); });
  setTimeout(function () { var r = document.getElementById('root'); send('probe: root children=' + (r ? r.childElementCount : -1) + ' visible=' + document.visibilityState); }, 4000);
})();
</script>`,
    ),
});

// Served to the Tauri webview; the content security policy lives in src-tauri/tauri.conf.json.
export default defineConfig({
  plugins: [react(), devProbe()],
  base: './',
  clearScreen: false,
  server: { port: 5173, strictPort: true, watch: { ignored: ['**/src-tauri/**', '**/engine/**'] } },
  build: { outDir: 'dist', emptyOutDir: true, target: ['es2021', 'safari15', 'chrome105'] },
});
