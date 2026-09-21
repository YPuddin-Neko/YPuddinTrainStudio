import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { readFileSync } from 'node:fs';
import packageJson from './package.json';
import { buildFingerprint } from './buildFingerprint.ts';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react(), buildFingerprint(), {
    name: 'local-test-worker',
    configureServer(server) {
      server.middlewares.use('/mockServiceWorker.js', (_req, res) => {
        res.setHeader('Content-Type', 'application/javascript');
        res.end(readFileSync(new URL('../Test/frontend/public/mockServiceWorker.js', import.meta.url)));
      });
    },
  }],
  resolve: { dedupe: [...Object.keys(packageJson.dependencies), ...Object.keys(packageJson.devDependencies)] },
  server: {
    port: 3000,
    fs: { allow: ['..'] },
    proxy: {
      '/api': {
        target: loadEnv(mode, '.', '').VITE_BACKEND_URL || 'http://127.0.0.1:8123',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // echarts 按需引入后仍 ~550 kB（图表引擎本体），已单独分块且仅 JobDetail 懒加载时拉取，放宽该块的告警线
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes('node_modules/echarts') || id.includes('node_modules/zrender')) return 'echarts';
          if (
            /node_modules\/(react|react-dom|react-router|@tanstack|i18next|lucide-react|scheduler)\//.test(id)
          ) {
            return 'vendor';
          }
        },
      },
    },
  },
}));
