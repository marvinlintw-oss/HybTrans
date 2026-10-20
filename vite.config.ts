// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: './',
  plugins: [react()],
  optimizeDeps: {
    // 關鍵：將 Worker 會用到的 @huggingface/transformers 預先列入 include，
    // 徹底阻止首次上傳音檔啟動 Worker 時觸發 Vite "re-optimizing dependencies" 自動 Refresh！
    include: [
      '@huggingface/transformers',
      'opencc-js',
      'lucide-react',
      'zustand',
      'mammoth',
      'jszip',
    ],
    exclude: ['@mlc-ai/web-llm'],
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