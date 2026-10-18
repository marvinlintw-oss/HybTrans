// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: './', // 關鍵：確保在 GitHub Pages (/HybTrans/) 與本機皆能正確解析資源與 Worker 路徑
  plugins: [react()],
  optimizeDeps: {
    exclude: ['onnxruntime-web', '@mlc-ai/web-llm'],
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
    chunkSizeWarningLimit: 3000,
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
});