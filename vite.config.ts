import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  // 設定相對基底路徑 './'，確保部署至 https://marvinlintw-oss.github.io/HybTrans/ 時，
  // 打包產出的 index.html 與 Worker 會以 ./assets/... 相對路徑載入資源，與 /HybTransPlus/ 完全獨立互不干擾。
  base: './',
  plugins: [react()],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // diarization.worker.ts 內使用了動態載入 import('onnxruntime-web')，
    // 預先列入 include 可避免 Vite 在首次上傳音檔啟動 Worker 時觸發頁面強制 Reload。
    include: [
      '@huggingface/transformers',
      'onnxruntime-web',
      'opencc-js',
      'lucide-react',
      'zustand',
      'mammoth',
      'jszip',
    ],
    exclude: ['@mlc-ai/web-llm'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});