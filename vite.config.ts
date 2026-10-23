import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // 關鍵修復：diarization.worker.ts 內使用了動態載入 import('onnxruntime-web')，
    // 若未將 onnxruntime-web 預先列入 include，Vite 會在第一次上傳音檔啟動 Worker 時才發現該套件，
    // 進而觸發 "optimized dependencies changed. reloading" 強制重整整個網頁！
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