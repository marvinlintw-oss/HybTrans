import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  // 關鍵修復：設定相對基底路徑 './'，確保部署至 GitHub Pages 子路徑 (如 /HybTransPlus/) 時，
  // 打包產出的 index.html 會以 ./assets/... 相對路徑載入 JS/CSS，避免指向根目錄 /assets/... 導致 404 白畫面！
  base: './',
  plugins: [react()],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // 關鍵鐵律：diarization.worker.ts 內使用了動態載入 import('onnxruntime-web')，
    // 必須預先列入 include，避免 Vite 在首次上傳音檔啟動 Worker 時觸發頁面強制 Reload。
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