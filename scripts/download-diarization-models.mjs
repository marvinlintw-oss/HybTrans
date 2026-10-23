#!/usr/bin/env node
/**
 * scripts/download-diarization-models.mjs
 * HybTransPlus 次世代語者分離自託管模型一鍵下載與驗證工具 (MIT / Apache-2.0 開源合規權重)
 *
 * 使用方式：
 *   node scripts/download-diarization-models.mjs
 *
 * 下載目標目錄：
 *   public/models/
 *     ├── silero_vad_v5.onnx                    (~2.0 MB, Silero VAD v5, MIT License)
 *     ├── segmentation-3.0.onnx                 (~5.7 MB, Pyannote Segmentation 3.0 Powerset EEND, MIT License)
 *     ├── plda-params-vbx.json                  (~1.0 MB, WeSpeaker/Pyannote PLDA + VBx 投影轉換參數)
 *     └── wespeaker_voxceleb_resnet34_int8.onnx (~26 MB, 若已存在則自動跳過)
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');
const TARGET_DIR = path.join(PROJECT_ROOT, 'public', 'models');

const MODEL_MANIFEST = [
  {
    filename: 'silero_vad_v5.onnx',
    description: 'Silero VAD v5 (16kHz 32ms 串流語音活性檢測與 12-22s 靜音谷底最佳化切分)',
    license: 'MIT',
    minSizeBytes: 1_500_000,
    urls: [
      'https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx',
      'https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx',
    ],
  },
  {
    filename: 'segmentation-3.0.onnx',
    description: 'Pyannote Segmentation 3.0 (10s 視窗 / 7 類別 Powerset 局部多語者與重疊語音分割)',
    license: 'MIT',
    minSizeBytes: 5_000_000,
    urls: [
      'https://huggingface.co/briox/diarization-js-community-1/resolve/main/segmentation-3.0.onnx',
      'https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/main/onnx/model.onnx',
      'https://huggingface.co/shareed2k/callmind-pyannote-segmentation/resolve/main/model.onnx',
    ],
  },
  {
    filename: 'plda-params-vbx.json',
    description: 'PLDA + VBx 變分貝氏全域聚類預訓練轉換矩陣 (256-dim WeSpeaker -> 128-dim PLDA 空間)',
    license: 'Apache-2.0 / MIT',
    minSizeBytes: 100_000,
    urls: [
      'https://huggingface.co/briox/diarization-js-community-1/resolve/main/plda-params-vbx.json',
    ],
  },
];

function computeSha256(filePath) {
  const hash = crypto.createHash('sha256');
  const data = fs.readFileSync(filePath);
  hash.update(data);
  return hash.digest('hex');
}

async function downloadWithFallback(item) {
  const destPath = path.join(TARGET_DIR, item.filename);

  if (fs.existsSync(destPath)) {
    const stat = fs.statSync(destPath);
    if (stat.size >= item.minSizeBytes) {
      const sha256 = computeSha256(destPath);
      console.log(
        `✅ [已存在] ${item.filename} (${(stat.size / (1024 * 1024)).toFixed(2)} MB | SHA256: ${sha256.slice(0, 12)}...)`
      );
      return true;
    }
  }

  console.log(`\n⬇️  正在下載 ${item.filename} — ${item.description}...`);

  for (const url of item.urls) {
    try {
      console.log(`   嘗試來源: ${url}`);
      const res = await fetch(url, {
        headers: { 'User-Agent': 'HybTransPlus-Model-Downloader/1.0' },
        redirect: 'follow',
      });

      if (!res.ok) {
        console.warn(`   ⚠️ HTTP ${res.status} (${res.statusText})，切換下一個鏡像源...`);
        continue;
      }

      const arrayBuffer = await res.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);

      if (buffer.byteLength < item.minSizeBytes) {
        console.warn(
          `   ⚠️ 下載檔案過小 (${buffer.byteLength} bytes < 預期 ${item.minSizeBytes} bytes)，可能為 LFS 指標，切換下一個鏡像源...`
        );
        continue;
      }

      fs.writeFileSync(destPath, buffer);
      const sha256 = computeSha256(destPath);
      console.log(
        `✅ [下載成功] ${item.filename} (${(buffer.byteLength / (1024 * 1024)).toFixed(2)} MB | SHA256: ${sha256.slice(0, 16)}...)`
      );
      return true;
    } catch (err) {
      console.warn(`   ⚠️ 連線失敗 (${err.message})，切換下一個鏡像源...`);
    }
  }

  console.warn(
    `⚠️ [無法下載] ${item.filename}：所有遠端來源皆無法連線。系統在執行期將自動啟動「純前端改良版能量 VAD + E2CP 受限譜聚類」Fallback 模式，不影響正常轉寫！`
  );
  return false;
}

async function main() {
  console.log('==============================================================');
  console.log('🚀 HybTransPlus 次世代語者分離本地 ONNX 模型與權重部署腳本');
  console.log(`📂 目標路徑: ${TARGET_DIR}`);
  console.log('==============================================================');

  if (!fs.existsSync(TARGET_DIR)) {
    fs.mkdirSync(TARGET_DIR, { recursive: true });
  }

  let successCount = 0;
  for (const item of MODEL_MANIFEST) {
    const ok = await downloadWithFallback(item);
    if (ok) successCount++;
  }

  console.log('\n==============================================================');
  console.log(
    `🎉 部署完成：成功就緒 ${successCount} / ${MODEL_MANIFEST.length} 個擴充模型檔案。`
  );
  console.log(
    '💡 提示：即使部分檔案未下載，diarization.worker.ts 已內建無縫降級機制（自動退回無 162Hz 硬切之 E2CP 受限譜聚類與自適應能量谷底切窗），絕不拋錯崩潰。'
  );
  console.log('==============================================================');
}

main().catch((err) => {
  console.error('❌ 模型下載腳本執行異常:', err);
  process.exit(1);
});