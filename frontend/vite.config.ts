import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import packageJson from './package.json' with { type: 'json' };
import { buildFingerprint } from './buildFingerprint.ts';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react(), buildFingerprint()],
  resolve: { dedupe: [...Object.keys(packageJson.dependencies), ...Object.keys(packageJson.devDependencies)] },
  server: {
    port: 3000,
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
