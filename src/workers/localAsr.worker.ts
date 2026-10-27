// src/workers/localAsr.worker.ts
// 次世代本地 WebGPU Whisper 語音轉寫與聲學換人錨點耦合引擎（q8 高精度聲學編碼器 + 防 OOM 低記憶體峰值序列載入版）
// 核心升級與根因修正：
// 1. [兼顧高精度殘響環境解析力與 16GB RAM 零 OOM 之黃金混合量化 (`encoder_model: 'q8'` + `decoder: 'q4f16'`)]：
//    - 針對 4-bit Encoder (`q4f16`) 在殘響會議室與遠距麥克風下容易壓縮頻譜注意力動態範圍（把「公有空間」聽成「風天/罐空間」、「32處」聽成「三個處」）之問題，
//    - 在標準與高精細模式下將 `encoder_model` 升級為 8-bit 對稱量化 (`q8` = 645MB，僅為 1.27GB `fp16` 的一半，且保留完整 80-bin 梅爾頻譜注意力精度)，搭配 `decoder_model_merged: 'q4f16' / 'q4'` (194MB)，總權重僅 839MB；效能安全模式 (`eco_safe`) 則維持 564MB `q4f16`。
// 2. [根除 `Unable to determine content-length` 導致之 1GB->2GB Uint8Array 動態擴張崩潰]：
//    - 攔截 Worker 內之 `fetch` 與 `Cache.prototype.match`，當回應缺少 `Content-Length` 標頭時，以每 24MB 自動壓實之磁碟級 `Blob` 串流接收並補齊精確 `Content-Length`。
// 3. [Encoder 與 Decoder 序列載入互斥鎖 (Sequential ONNX Init Mutex)]：
//    - 強制讓 `encoder_model` 完成載入與釋放後才啟動 `decoder_model_merged` 串流，打破雙模型並發解壓縮峰值。
// 4. [強化聲學換人切點子句切分 (`splitClauseAtAcousticSwitches`)]：
//    - 當 Whisper 將兩人快速問答連寫為同一句時，允許在聲學話輪切換點 (`midClauseSwitches`) 附近之自然詞界（含無逗號處）切開並標記 `isAcousticTurnBoundary = true`。

import { env, pipeline } from '@huggingface/transformers';
import * as OpenCC from 'opencc-js';
import {
  FineDiarizationTurn,
  FrameSpeakerPosteriorMatrix,
  GranularityTier,
  LocalAsrModel,
  Segment,
  VirtualTrackMapItem,
  WordSpan,
} from '../types';
import {
  decodeWordLevelSpeakersViterbi,
  groupAlignedWordsIntoSegments,
  RawWordTimestampItem,
  segmentNaturalLanguageTokens,
  synthesizePosteriorMatrixFromFineTurns,
} from '../utils/viterbiWordAligner';

env.allowLocalModels = false;
env.useBrowserCache = true;

if (env.backends?.onnx) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (env.backends.onnx as any).logLevel = 'error';
  if (env.backends.onnx.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
    env.backends.onnx.wasm.proxy = false;
  }
}

const toTw = OpenCC.Converter({ from: 'cn', to: 'tw' });

const MODEL_ID_MAP: Record<LocalAsrModel, string> = {
  'Whisper-Large-v3-Turbo': 'onnx-community/whisper-large-v3-turbo',
  'whisper-large-v3-turbo': 'onnx-community/whisper-large-v3-turbo',
  'SenseVoice-Small': 'onnx-community/whisper-small',
  'whisper-base': 'onnx-community/whisper-base',
  'Breeze-ASR': 'onnx-community/whisper-base',
  'breeze-asr-25': 'onnx-community/whisper-base',
};

const DEFAULT_ALIGNMENT_HEADS: Record<LocalAsrModel, number[][]> = {
  'Whisper-Large-v3-Turbo': [
    [2, 4],
    [2, 11],
    [3, 3],
    [3, 6],
    [3, 11],
    [3, 14],
  ],
  'whisper-large-v3-turbo': [
    [2, 4],
    [2, 11],
    [3, 3],
    [3, 6],
    [3, 11],
    [3, 14],
  ],
  'SenseVoice-Small': [
    [5, 3],
    [5, 9],
    [8, 0],
    [8, 4],
    [8, 7],
    [8, 8],
    [9, 0],
    [9, 7],
    [9, 9],
    [10, 5],
  ],
  'whisper-base': [
    [3, 1],
    [4, 2],
    [4, 3],
    [4, 7],
    [5, 1],
    [5, 2],
    [5, 4],
    [5, 6],
  ],
  'Breeze-ASR': [
    [3, 1],
    [4, 2],
    [4, 3],
    [4, 7],
    [5, 1],
    [5, 2],
    [5, 4],
    [5, 6],
  ],
  'breeze-asr-25': [
    [3, 1],
    [4, 2],
    [4, 3],
    [4, 7],
    [5, 1],
    [5, 2],
    [5, 4],
    [5, 6],
  ],
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let asrInstance: any = null;
let loadedModelKey: string | null = null;
let maxReportedProgress = 0;
let activeDownloadingModelName: string = 'Whisper-Large-v3-Turbo';
let isIntegratedOrLowMemDevice = false;
const fileProgressMap = new Map<string, { loaded: number; total: number }>();

function estimateKnownOnnxFileBytes(urlStr: string): number {
  if (urlStr.includes('encoder_model_q4f16')) return 370 * 1024 * 1024;
  if (urlStr.includes('encoder_model_q4')) return 425 * 1024 * 1024;
  if (
    urlStr.includes('encoder_model_quantized') ||
    urlStr.includes('encoder_model_int8')
  ) {
    return 645 * 1024 * 1024;
  }
  if (urlStr.includes('encoder_model_fp16')) return 1270 * 1024 * 1024;
  if (urlStr.includes('decoder_model_merged_q4f16')) return 194 * 1024 * 1024;
  if (urlStr.includes('decoder_model_merged_q4')) return 334 * 1024 * 1024;
  if (urlStr.includes('decoder_model_merged_quantized')) return 440 * 1024 * 1024;
  return 250 * 1024 * 1024;
}

function reportUnifiedDownloadProgress(
  fileKey: string,
  loadedBytes: number,
  totalBytes: number
): void {
  const shortFile = fileKey.split('/').pop()?.split('?')[0] || fileKey;
  if (!shortFile.endsWith('.onnx')) return;

  fileProgressMap.set(shortFile, {
    loaded: loadedBytes,
    total: Math.max(loadedBytes, totalBytes),
  });

  let encoderRatio = 0;
  let decoderRatio = 0;
  let hasEncoder = false;
  let hasDecoder = false;

  for (const [k, v] of fileProgressMap.entries()) {
    const r = v.total > 0 ? Math.min(1.0, v.loaded / v.total) : 0;
    if (k.includes('encoder')) {
      hasEncoder = true;
      encoderRatio = Math.max(encoderRatio, r);
    } else if (k.includes('decoder')) {
      hasDecoder = true;
      decoderRatio = Math.max(decoderRatio, r);
    }
  }

  let combinedPct = 10;
  if (hasEncoder && hasDecoder) {
    combinedPct = Math.round((encoderRatio * 0.68 + decoderRatio * 0.32) * 96);
  } else if (hasEncoder) {
    combinedPct = Math.round(encoderRatio * 66);
  } else if (hasDecoder) {
    combinedPct = 66 + Math.round(decoderRatio * 32);
  }

  maxReportedProgress = Math.max(
    maxReportedProgress,
    Math.min(99, Math.max(6, combinedPct))
  );

  self.postMessage({
    type: 'ASR_DOWNLOAD_PROGRESS',
    active: true,
    modelName: activeDownloadingModelName,
    fileName: shortFile,
    progress: maxReportedProgress,
  });
}

let onnxStreamLockPromise: Promise<void> = Promise.resolve();

function wrapStreamWithReleaseHook(
  sourceStream: ReadableStream<Uint8Array>,
  onComplete: () => void
): ReadableStream<Uint8Array> {
  const reader = sourceStream.getReader();
  let released = false;
  const triggerRelease = () => {
    if (!released) {
      released = true;
      setTimeout(onComplete, 320);
    }
  };

  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          controller.close();
          triggerRelease();
          return;
        }
        if (value) {
          controller.enqueue(value);
        }
      } catch (err) {
        triggerRelease();
        controller.error(err);
      }
    },
    cancel(reason) {
      triggerRelease();
      return reader.cancel(reason);
    },
  });
}

let interceptorsInstalled = false;

function installMemorySafeFetchAndCacheInterceptor(): void {
  if (interceptorsInstalled) return;
  interceptorsInstalled = true;

  if (typeof Cache !== 'undefined' && Cache.prototype?.match) {
    const origMatch = Cache.prototype.match;
    Cache.prototype.match = async function (
      request: RequestInfo | URL,
      options?: CacheQueryOptions
    ): Promise<Response | undefined> {
      const matched = await origMatch.call(this, request, options);
      if (!matched) return undefined;

      const reqUrl =
        typeof request === 'string'
          ? request
          : request instanceof URL
          ? request.href
          : request.url || '';
      const isOnnx = reqUrl.includes('.onnx');

      let releaseLock: () => void = () => {};
      if (isOnnx) {
        const prevLock = onnxStreamLockPromise;
        onnxStreamLockPromise = new Promise<void>((resolve) => {
          releaseLock = resolve;
        });
        await prevLock;
      }

      try {
        const diskBlob = await matched.blob();
        const exactSize = diskBlob.size;
        const headers = new Headers(matched.headers);
        headers.set('Content-Length', String(exactSize));

        if (isOnnx) {
          reportUnifiedDownloadProgress(reqUrl, exactSize, exactSize);
          const hookedStream = wrapStreamWithReleaseHook(
            diskBlob.stream(),
            releaseLock
          );
          return new Response(hookedStream, {
            status: matched.status || 200,
            statusText: matched.statusText || 'OK',
            headers,
          });
        }

        return new Response(diskBlob.stream(), {
          status: matched.status || 200,
          statusText: matched.statusText || 'OK',
          headers,
        });
      } catch {
        if (isOnnx) releaseLock();
        return matched;
      }
    };
  }

  const origFetch = self.fetch.bind(self);
  self.fetch = async function (
    input: RequestInfo | URL,
    init?: RequestInit
  ): Promise<Response> {
    const reqUrl =
      typeof input === 'string'
        ? input
        : input instanceof URL
        ? input.href
        : input.url || '';
    const isOnnx = reqUrl.includes('.onnx');

    let releaseLock: () => void = () => {};
    if (isOnnx) {
      const prevLock = onnxStreamLockPromise;
      onnxStreamLockPromise = new Promise<void>((resolve) => {
        releaseLock = resolve;
      });
      await prevLock;
    }

    try {
      const res = await origFetch(input, init);
      if (!res.ok || !res.body) {
        if (isOnnx) releaseLock();
        return res;
      }

      const rawContentLen =
        res.headers.get('content-length') ||
        res.headers.get('x-linked-size') ||
        '';
      const parsedLen = parseInt(rawContentLen, 10);

      if (Number.isFinite(parsedLen) && parsedLen > 0) {
        const headers = new Headers(res.headers);
        headers.set('Content-Length', String(parsedLen));
        if (isOnnx) {
          const hookedStream = wrapStreamWithReleaseHook(res.body, releaseLock);
          return new Response(hookedStream, {
            status: res.status,
            statusText: res.statusText,
            headers,
          });
        }
        return new Response(res.body, {
          status: res.status,
          statusText: res.statusText,
          headers,
        });
      }

      if (!isOnnx) {
        const buf = await res.arrayBuffer();
        const headers = new Headers(res.headers);
        headers.set('Content-Length', String(buf.byteLength));
        return new Response(buf, {
          status: res.status,
          statusText: res.statusText,
          headers,
        });
      }

      const estimatedTotal = estimateKnownOnnxFileBytes(reqUrl);
      const reader = res.body.getReader();
      let compactedBlob = new Blob([]);
      let chunkBatch: BlobPart[] = [];
      let batchBytes = 0;
      let totalLoaded = 0;
      const COMPACT_THRESHOLD_BYTES = 24 * 1024 * 1024;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value && value.byteLength > 0) {
          const sliceCopy = value.buffer.slice(
            value.byteOffset,
            value.byteOffset + value.byteLength
          );
          chunkBatch.push(sliceCopy);
          batchBytes += value.byteLength;
          totalLoaded += value.byteLength;

          if (batchBytes >= COMPACT_THRESHOLD_BYTES) {
            compactedBlob = new Blob([compactedBlob, ...chunkBatch], {
              type: 'application/octet-stream',
            });
            chunkBatch = [];
            batchBytes = 0;
          }

          reportUnifiedDownloadProgress(
            reqUrl,
            totalLoaded,
            Math.max(totalLoaded + 1024 * 1024, estimatedTotal)
          );
        }
      }

      if (chunkBatch.length > 0) {
        compactedBlob = new Blob([compactedBlob, ...chunkBatch], {
          type: 'application/octet-stream',
        });
        chunkBatch = [];
        batchBytes = 0;
      }

      reportUnifiedDownloadProgress(
        reqUrl,
        compactedBlob.size,
        compactedBlob.size
      );

      const headers = new Headers(res.headers);
      headers.set('Content-Length', String(compactedBlob.size));
      headers.set('Content-Type', 'application/octet-stream');

      const hookedStream = wrapStreamWithReleaseHook(
        compactedBlob.stream(),
        releaseLock
      );
      return new Response(hookedStream, {
        status: res.status,
        statusText: res.statusText,
        headers,
      });
    } catch (err) {
      if (isOnnx) releaseLock();
      throw err;
    }
  };
}

installMemorySafeFetchAndCacheInterceptor();

async function purgeOversizedLegacyModelCache(): Promise<void> {
  if (!('caches' in self)) return;
  try {
    const cache = await caches.open('transformers-cache');
    const keys = await cache.keys();
    for (const req of keys) {
      if (
        req.url.includes('whisper-large-v3-turbo') &&
        (req.url.endsWith('/encoder_model_fp16.onnx') ||
          req.url.endsWith('/encoder_model.onnx') ||
          req.url.endsWith('/encoder_model.onnx_data'))
      ) {
        await cache.delete(req);
      }
    }
  } catch {
    // ignore cache cleanup error
  }
}

async function probeWebGpuCapabilities(): Promise<{
  hasWebGpu: boolean;
  supportsFp16: boolean;
  isIntegratedGpu: boolean;
  maxBufferSizeMB: number;
}> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as unknown as { gpu?: any }).gpu;
    if (!gpu || typeof gpu.requestAdapter !== 'function') {
      return {
        hasWebGpu: false,
        supportsFp16: false,
        isIntegratedGpu: true,
        maxBufferSizeMB: 0,
      };
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      return {
        hasWebGpu: false,
        supportsFp16: false,
        isIntegratedGpu: true,
        maxBufferSizeMB: 0,
      };
    }
    const supportsFp16 = Boolean(
      adapter.features &&
        typeof adapter.features.has === 'function' &&
        adapter.features.has('shader-f16')
    );

    const info =
      (
        adapter as unknown as {
          info?: {
            vendor?: string;
            architecture?: string;
            description?: string;
          };
        }
      ).info || {};
    const vendor = (info.vendor || '').toLowerCase();
    const desc = (
      info.description ||
      info.architecture ||
      ''
    ).toLowerCase();

    const maxBufferBytes = adapter.limits?.maxBufferSize || 268435456;
    const maxStorageBytes =
      adapter.limits?.maxStorageBufferBindingSize || 134217728;
    const maxBufferSizeMB = Math.round(
      Math.max(maxBufferBytes, maxStorageBytes) / (1024 * 1024)
    );

    const isIntegratedKeyword =
      desc.includes('uhd') ||
      desc.includes('iris') ||
      desc.includes('hd graphics') ||
      (vendor.includes('intel') && !desc.includes('arc')) ||
      (vendor.includes('amd') && desc.includes('radeon(tm) graphics'));

    const isDiscreteKeyword =
      vendor.includes('nvidia') ||
      desc.includes('rtx') ||
      desc.includes('gtx') ||
      desc.includes('radeon rx') ||
      desc.includes('arc');

    const isIntegratedGpu =
      isIntegratedKeyword || (!isDiscreteKeyword && maxBufferSizeMB < 2048);

    return {
      hasWebGpu: true,
      supportsFp16,
      isIntegratedGpu,
      maxBufferSizeMB,
    };
  } catch {
    return {
      hasWebGpu: false,
      supportsFp16: false,
      isIntegratedGpu: true,
      maxBufferSizeMB: 0,
    };
  }
}

function ensureAlignmentHeadsPatched(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transcriber: any,
  modelName: LocalAsrModel
): void {
  try {
    const genCfg = transcriber?.model?.generation_config;
    if (
      genCfg &&
      (!Array.isArray(genCfg.alignment_heads) ||
        genCfg.alignment_heads.length === 0)
    ) {
      genCfg.alignment_heads =
        DEFAULT_ALIGNMENT_HEADS[modelName] ||
        DEFAULT_ALIGNMENT_HEADS['Whisper-Large-v3-Turbo'];
    }
  } catch {
    // ignore
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function releaseResidualGpuTensors(outputObj: any): void {
  try {
    if (!outputObj || typeof outputObj !== 'object') return;
    for (const key of Object.keys(outputObj)) {
      const val = outputObj[key];
      if (val && typeof val.dispose === 'function') {
        val.dispose();
      }
    }
  } catch {
    // ignore
  }
}

async function ensureAsrPipeline(
  modelName: LocalAsrModel,
  granularityTier: GranularityTier = 'balanced_pro'
) {
  const modelId =
    MODEL_ID_MAP[modelName] || MODEL_ID_MAP['Whisper-Large-v3-Turbo'];
  const cacheKey = `${modelId}::${granularityTier}`;
  if (asrInstance && loadedModelKey === cacheKey) {
    ensureAlignmentHeadsPatched(asrInstance, modelName);
    return asrInstance;
  }

  if (asrInstance && typeof asrInstance.dispose === 'function') {
    try {
      await asrInstance.dispose();
    } catch {
      // ignore
    }
    asrInstance = null;
  }

  await purgeOversizedLegacyModelCache();

  maxReportedProgress = 5;
  activeDownloadingModelName = modelName;
  fileProgressMap.clear();

  self.postMessage({
    type: 'ASR_DOWNLOAD_PROGRESS',
    active: true,
    modelName,
    fileName: '正在檢測硬體規格並準備高精度語音模型...',
    progress: 5,
  });

  const progressCallback = (info: {
    status?: string;
    file?: string;
    loaded?: number;
    total?: number;
    progress?: number;
  }) => {
    const fileName = info.file || '';
    if (!fileName.endsWith('.onnx')) return;

    if (info.status === 'progress' || info.status === 'download') {
      if (
        typeof info.loaded === 'number' &&
        typeof info.total === 'number' &&
        info.total > 0
      ) {
        reportUnifiedDownloadProgress(fileName, info.loaded, info.total);
      } else if (typeof info.progress === 'number') {
        maxReportedProgress = Math.max(
          maxReportedProgress,
          Math.min(99, Math.round(info.progress))
        );
        const shortFile = fileName.split('/').pop() || fileName;
        self.postMessage({
          type: 'ASR_DOWNLOAD_PROGRESS',
          active: true,
          modelName,
          fileName: shortFile,
          progress: maxReportedProgress,
        });
      }
    }
  };

  const { hasWebGpu, supportsFp16, isIntegratedGpu, maxBufferSizeMB } =
    await probeWebGpuCapabilities();
  isIntegratedOrLowMemDevice =
    isIntegratedGpu || maxBufferSizeMB < 2048 || granularityTier === 'eco_safe';

  // 【黃金混合量化策略】：
  // - 標準平衡 (`balanced_pro`) 與高精細模式 (`ultra_studio`)：Encoder 採用 8-bit `q8` (645MB，保留完整頻譜注意力精度，解決殘響環境把「公有空間」聽成「風天」之失真)，Decoder 採用 `q4f16 / q4` (194MB)，總計 839MB，兼顧零 OOM 與高辨識率！
  // - 效能安全模式 (`eco_safe`) 或極低顯存裝置 (< 768MB)：Encoder 與 Decoder 皆採用 `q4f16 / q4` (564MB)。
  const useHighPrecisionQ8Encoder =
    granularityTier !== 'eco_safe' && maxBufferSizeMB >= 768;

  const preferredWebGpuDtype = useHighPrecisionQ8Encoder
    ? ({
        encoder_model: 'q8',
        decoder_model_merged: supportsFp16 ? 'q4f16' : 'q4',
      } as const)
    : supportsFp16
    ? ({
        encoder_model: 'q4f16',
        decoder_model_merged: 'q4f16',
      } as const)
    : ({
        encoder_model: 'q4',
        decoder_model_merged: 'q4',
      } as const);

  try {
    if (hasWebGpu) {
      if (env.backends?.onnx?.wasm) {
        env.backends.onnx.wasm.numThreads = 1;
      }
      console.info(
        `ℹ️ [LocalASR] 啟動 WebGPU 推論 (${
          isIntegratedGpu ? '整合式內顯共享記憶體保護模式' : '獨立顯卡高速模式'
        }, dtype=${preferredWebGpuDtype.encoder_model}/${
          preferredWebGpuDtype.decoder_model_merged
        })`
      );
      asrInstance = await pipeline('automatic-speech-recognition', modelId, {
        device: 'webgpu',
        dtype: preferredWebGpuDtype,
        progress_callback: progressCallback,
      });
    } else {
      throw new Error('WebGPU unavailable, fallback to WASM');
    }
  } catch (firstErr) {
    console.warn(
      '⚠️ [LocalASR] 首選 WebGPU 模式切換至相容保護模式 (q4f16 / q4 / WASM):',
      firstErr
    );
    fileProgressMap.clear();
    try {
      if (hasWebGpu) {
        asrInstance = await pipeline('automatic-speech-recognition', modelId, {
          device: 'webgpu',
          dtype: {
            encoder_model: supportsFp16 ? 'q4f16' : 'q4',
            decoder_model_merged: supportsFp16 ? 'q4f16' : 'q4',
          },
          progress_callback: progressCallback,
        });
      } else {
        if (env.backends?.onnx?.wasm) {
          env.backends.onnx.wasm.numThreads = 2;
        }
        asrInstance = await pipeline('automatic-speech-recognition', modelId, {
          device: 'wasm',
          dtype: 'q8',
          progress_callback: progressCallback,
        });
      }
    } catch (secondErr) {
      console.warn(
        '⚠️ [LocalASR] 偵測到極端記憶體受限環境，自動啟動輕量防崩潰模型 (whisper-small q8 ~160MB):',
        secondErr
      );
      fileProgressMap.clear();
      if (env.backends?.onnx?.wasm) {
        env.backends.onnx.wasm.numThreads = 1;
      }
      const fallbackSafeModelId = 'onnx-community/whisper-small';
      asrInstance = await pipeline(
        'automatic-speech-recognition',
        fallbackSafeModelId,
        {
          device: hasWebGpu ? 'webgpu' : 'wasm',
          dtype: hasWebGpu
            ? {
                encoder_model: 'q4',
                decoder_model_merged: 'q4',
              }
            : 'q8',
          progress_callback: progressCallback,
        }
      );
    }
  }

  ensureAlignmentHeadsPatched(asrInstance, modelName);

  self.postMessage({
    type: 'ASR_DOWNLOAD_PROGRESS',
    active: false,
    modelName,
    fileName: '',
    progress: 100,
  });

  loadedModelKey = cacheKey;
  return asrInstance;
}

function estimateRepetitionCompressionRatio(text: string): number {
  const clean = text.replace(/\s+/g, '').trim();
  if (clean.length < 8) return 1.0;

  if (
    /([\u4e00-\u9fa5a-zA-Z0-9]{1,6})(?:[，。？！、；：,.?!\s]*\1){3,}/i.test(
      clean
    )
  ) {
    return 3.2;
  }

  const pureChars = clean.replace(/[，。？！、；：,.?!]/g, '');
  if (pureChars.length < 8) return 1.0;

  const bigrams = new Set<string>();
  for (let i = 0; i < pureChars.length - 1; i++) {
    bigrams.add(pureChars.slice(i, i + 2));
  }
  const uniqueRatio = bigrams.size / Math.max(1, pureChars.length - 1);
  return Number((1.0 / Math.max(0.15, uniqueRatio)).toFixed(2));
}

function restoreMultilingualPunctuation(
  rawInput: string,
  options?: {
    isMidTurnContinuation?: boolean;
    nextChunkRawText?: string;
    pauseToNextSec?: number;
  }
): string {
  const hasKana = /[\u3040-\u30ff]/.test(rawInput);
  let text = hasKana ? rawInput.trim() : toTw(rawInput.trim());
  if (!text) return '';

  text = text.replace(/([\u4e00-\u9fa5])\1{2,}/g, '$1');
  text = text.replace(/([\u4e00-\u9fa5]{2,6})\1{2,}/g, '$1');
  text = text.replace(
    /([\u4e00-\u9fa5]{1,6})(?:[，。？！、；：,.?!\s]+\1){2,}/g,
    '$1'
  );
  text = text.replace(/^[常嗯恩啊喔欸][。，、,.]\s*(?=謝謝|好|對|可以)/, '');
  text = text.replace(
    /^(字幕|請不吝點讚|訂閱轉發|謝謝觀看|Amara\.org).*$/gi,
    ''
  );

  const cjkRatio =
    (text.match(/[\u3040-\u9fa5]/g)?.length || 0) / Math.max(1, text.length);

  if (cjkRatio >= 0.25) {
    text = text
      .replace(/,/g, '，')
      .replace(/\?/g, '？')
      .replace(/!/g, '！');

    text = text.replace(
      /([\u3040-\u9fa5]+)\s+([\u3040-\u9fa5]+)/g,
      (_match, leftChunk: string, rightChunk: string) => {
        if (isSyntacticallyInseparableBoundary(leftChunk, rightChunk)) {
          return `${leftChunk}${rightChunk}`;
        }
        const isShortStandaloneInterjection =
          /^(對|好|是|不是|沒有|算|不算|沒錯|可以|對啊|好啊|好那|那|有|喂)$/.test(
            leftChunk
          );
        if (isShortStandaloneInterjection) {
          return `${leftChunk}，${rightChunk}`;
        }
        if (leftChunk.length <= 2) {
          return `${leftChunk}${rightChunk}`;
        }
        return `${leftChunk}，${rightChunk}`;
      }
    );
  }

  text = text.replace(
    /^(?:(對|好)(?![於長官貴賓同仁講者團隊他她你我您大家這那外內接齊應照比面方手話象嗎呢吧啊呀喔啦])|(不是)(?![嗎呢說要很太都也因為])|(沒有)(?![嗎呢啊吧喔啦要人辦法問題關係錯想說那這])|(那)(?![個些邊裡麼天位種家間場樣一二三四五六七八九十百千萬兩嗎呢吧原])|(另外)(?![一二三四五六七八九十兩半個種邊家位])|(不過|其實|老實說|基本上|はい|ええ|いや|なるほど))(?=[\u3040-\u9fa5])/g,
    '$&，'
  );
  text = text.replace(
    /^(對對對|沒有沒有|不用不用)(?=[^\u3000-\u303f\uff00-\uffef])/g,
    '$1，'
  );

  text = text
    .replace(/那，(個|些|天|邊|裡|種|位|樣)/g, '那$1')
    .replace(/沒有，(嗎|呢|啊|吧|喔|要|問題|關係|辦法)/g, '沒有$1')
    .replace(/另外，(一[個種邊家位半])/g, '另外$1')
    .replace(/對，(長官|貴賓|同仁|講者|團隊|於|他|她|你|我|大家|口)/g, '對$1');

  text = text.replace(
    /([\u4e00-\u9fa5A-Za-z0-9]{4,})(但是|可是|不過|因為|所以|如果說|不然的話|否則|那如果|而且|然後|ですが|けれども|しかし|ただ)/g,
    '$1，$2'
  );

  text = text.replace(
    /(的時候|的話|來說|的部分|這件事)(?=[\u4e00-\u9fa5]{3,})/g,
    '$1，'
  );

  text = text.replace(
    /(怎麼辦|怎麼樣|怎樣|對不對|是不是|好不好|可以嗎|是嗎|對嗎|多少錢|為什麼|哪裡|什麼時候|甚麼時候|什麼意思|甚麼意思|誰會補助|接觸了沒|でしょうか|ますか|ですか)(?=[，。？?]|$)/g,
    '$1？'
  );
  text = text.replace(/(嗎|呢)[。，？?]*$/g, '$1？');

  text = text
    .replace(/[？?]{2,}/g, '？')
    .replace(/[！!]{2,}/g, '！')
    .replace(/？[，。、]+/g, '？')
    .replace(/！[，。、]+/g, '！')
    .replace(/[，、]{2,}/g, '，')
    .replace(/，([。？！])/g, '$1');

  if (
    text.length >= 2 &&
    cjkRatio >= 0.25 &&
    !/[，。？！、；：.?!]$/.test(text)
  ) {
    const nextRaw = options?.nextChunkRawText
      ? toTw(options.nextChunkRawText.trim())
      : '';
    const isMidTurn = Boolean(options?.isMidTurnContinuation);
    const pauseSec = options?.pauseToNextSec ?? 99;

    const endsWithIncompleteHead =
      /(?:[的之比在跟對向把被讓給為從到]|那個|這個|一些|一個|我們有|他們有)$/.test(
        text
      ) ||
      (nextRaw.length > 0 && isSyntacticallyInseparableBoundary(text, nextRaw));
    const endsWithClauseBridge =
      /(?:要能夠|能夠|可以|應該|就是|因為|所以|而且|然後|例如說|比方說|像是|主要是|還有|目前的話)$/.test(
        text
      );

    if (isMidTurn && endsWithIncompleteHead) {
      // 保持無標點，讓下一個 sub-chunk 自然無縫接上
    } else if (
      isMidTurn &&
      (endsWithClauseBridge || (pauseSec <= 0.85 && text.length <= 24))
    ) {
      text += '，';
    } else {
      text += '。';
    }
  }
  return text;
}

const UNIVERSAL_CLAUSE_STARTER_REGEX =
  /^(但是|可是|不過|因為|所以|如果說|那如果|要是|不是[，、]|沒有[，、]|不用[，、]|不然|否則|其實|老實說|那我們|那你們|那我|所以我|目前的話|好報告|報告主席|現在只有|現在總共|第一個是|第二個是|另外一個|這應該|對[，、]?這|好[，、]?再來|所以要|這個也是|這個他們|也是|那原民會|有[，、]?接觸|接觸哪|什麼意思|甚麼意思|喂)/;

function isSyntacticallyInseparableBoundary(
  leftStr: string,
  rightStr: string
): boolean {
  const lClean = leftStr.replace(/[，。？！、；：,.?!\s]+$/g, '');
  const rClean = rightStr.replace(/^[，。？！、；：,.?!\s]+/g, '');
  if (!lClean || !rClean) return false;
  if (/[，。？！、；：,.?!]$/.test(leftStr.trim())) return false;

  if (
    /[給寄送交找問請跟對向讓叫幫替被把為愛看聽說告訴通知聯絡協助感謝謝謝陪帶花寫放留住用落]$/.test(
      lClean
    ) &&
    /^(我|你|您|他|她|它|我們|你們|他們|她們|大家|誰|自己|在哪|在哪裡|在那裡|在這裡|到哪|到哪裡|給誰|成什麼)/.test(
      rClean
    )
  ) {
    return true;
  }

  if (
    (/([的之]|錢|店|人|事|案子|部分|問題|時候|地方|範圍|時程|對象|規劃|經費|獎勵金|補助)$/.test(
      lClean
    ) &&
      /^(就是|就要|就只要|是|要|會|可以|應該|沒有|不能|不用|都|也|還|才|已經)/.test(
        rClean
      )) ||
    (/[的之]$/.test(lClean) && /^[\u4e00-\u9fa5]{1,4}/.test(rClean))
  ) {
    return true;
  }

  if (
    /([0-9一二三四五六七八九十百千萬兩]+(到|至|跟|或|～|~)?|[到至])$/.test(
      lClean
    ) &&
    /^([0-9一二三四五六七八九十百千萬兩]|到|至|分|分鐘|小時|天|個|月|年|萬|元|塊|人|家|間|處|案|號|成|倍)/.test(
      rClean
    )
  ) {
    return true;
  }

  if (
    /(很|太|更|最|超|蠻|挺|都|還|不|沒|別|真|非常|相當|特別|十分|比較|稍微|有點|已經)$/.test(
      lClean
    ) &&
    /^(好|對|棒|行|可以|錯|大|小|多|少|高|低|快|慢|遠|近|久|難|易|滿|亂|重要|清楚|可能|確定|喜歡|想要|需要|應該|願意)/.test(
      rClean
    )
  ) {
    return true;
  }

  if (
    /^(的|得|地|了|嗎|呢|吧|啊|呀|喔|哦|啦|囉|嘛|齁|吼|過|著|們)([，。？！、；：,.?!\s]|$)/.test(
      rClean
    )
  ) {
    return true;
  }

  if (
    /[一二三四五六七八九十百千萬兩這那某每各幾]$/.test(lClean) &&
    /^(個|位|名|次|件|種|些|點|處|家|間|筆|項|套|場|人|天|年|月|週|期|標|案|組)/.test(
      rClean
    )
  ) {
    return true;
  }

  if (
    /(現|明|今|去|後|前|一|商|業|招|標|補|助|獎|勵|翻|裝|修|創|群|廊|店|公|問|時)$/.test(
      lClean
    ) &&
    /^(在|年|天|組|師|署|處|標|金|修|生|聚|帶|面|家|司|題|程)/.test(rClean)
  ) {
    return true;
  }

  return false;
}

function splitTextIntoDiscourseClauses(cleanClause: string): string[] {
  let t = cleanClause;
  t = t.replace(/([。？！?!])/g, '$1|');
  t = t.replace(
    /([|。？！?!]\s*(?:不算|算|對|對啊|對對對|好|好的|好再來|可以|沒錯|沒有|不是|了解|有|はい|ええ|うん|なるほど|yeah|yes|no|right|sure|ok|okay)[，、,])\s*/gi,
    '$1|'
  );
  t = t.replace(
    /(^|[|，。？！?!]\s*)(不算|算|對|好|可以|沒有|不是)([，、,])\s*(?=(?:不算|算|對|好|不是|沒有|我們|那|所以|可是|但是|再來))/g,
    '$1$2$3|'
  );
  t = t.replace(
    /([\u4e00-\u9fa5]{4,}[，、,])\s*(?=(?:好|好的|好啊|好再來|對|對啊|對對對|可以|沒錯|算|不算|了解|ok|okay)[。？！?!|\s]*$)/gi,
    '$1|'
  );
  t = t.replace(
    /([\u4e00-\u9fa5]{4,}[，、,])\s*(?=不是[，]|可是|不過|但是|然而|那如果|不然|否則|這應該|對[，]?這|好[，]?再來|所以要|那原民會|接觸哪|什麼意思|甚麼意思|はい|しかし|ただ|but\s|however)/gi,
    '$1|'
  );

  return t
    .split('|')
    .map((p) => p.trim())
    .filter((p) => p.length > 0 && !/^[，。？！、,.?!]+$/.test(p));
}

function tokenizeClausePreventingTailOrphan(cText: string): string[] {
  const cleaned = segmentNaturalLanguageTokens(cText);

  if (cleaned.length >= 2) {
    const last = cleaned[cleaned.length - 1];
    const prev = cleaned[cleaned.length - 2];
    const lastPure = last.replace(/[，。？！、；：,.?!\s]/g, '');
    const prevPure = prev.replace(/[，。？！、；：,.?!\s]/g, '');
    if (
      lastPure.length === 1 &&
      /^[\u4e00-\u9fa5]$/.test(lastPure) &&
      !/[，。？！、；：,.?!]$/.test(prev) &&
      /^[\u4e00-\u9fa5]{1,3}$/.test(prevPure)
    ) {
      cleaned.splice(cleaned.length - 2, 2, prev + last);
    }
  }

  return cleaned;
}

function splitClauseAtAcousticSwitches(
  cText: string,
  cStart: number,
  cEnd: number,
  midClauseSwitches: number[]
): Array<{ text: string; start: number; end: number; isAcousticSplit: boolean }> {
  if (midClauseSwitches.length === 0 || cText.length < 7) {
    return [{ text: cText, start: cStart, end: cEnd, isAcousticSplit: false }];
  }
  const chars = Array.from(cText);
  const L = chars.length;
  const dur = Math.max(0.2, cEnd - cStart);

  // 建立自然詞界索引集合，供無標點換人切點優先對齊
  const naturalTokens = segmentNaturalLanguageTokens(cText);
  const tokenBoundarySet = new Set<number>();
  let accLen = 0;
  for (const nt of naturalTokens) {
    accLen += nt.length;
    tokenBoundarySet.add(accLen);
  }

  const splitPoints: Array<{ charIdx: number; timeSec: number }> = [];
  for (const st of midClauseSwitches) {
    const rel = Math.max(0.18, Math.min(0.82, (st - cStart) / dur));
    const rawIdx = Math.round(rel * L);
    let bestIdx = -1;
    let bestScore = -5.0;

    for (
      let idx = Math.max(3, rawIdx - 4);
      idx <= Math.min(L - 3, rawIdx + 4);
      idx++
    ) {
      const leftPart = chars.slice(0, idx).join('');
      const restStr = chars.slice(idx).join('');
      if (isSyntacticallyInseparableBoundary(leftPart, restStr)) {
        continue;
      }

      const distPen = Math.abs(idx - rawIdx) * 0.38;
      const prevChar = chars[idx - 1];
      let score = -distPen;
      if (/[，、；：,:;\s]/.test(prevChar)) {
        score += 3.5;
      } else if (UNIVERSAL_CLAUSE_STARTER_REGEX.test(restStr)) {
        score += 2.5;
      } else if (tokenBoundarySet.has(idx)) {
        score += 1.4;
      }
      if (score > bestScore) {
        bestScore = score;
        bestIdx = idx;
      }
    }
    if (bestIdx >= 3 && bestScore >= 0.45) {
      splitPoints.push({ charIdx: bestIdx, timeSec: st });
    }
  }

  splitPoints.sort((a, b) => a.charIdx - b.charIdx);
  const result: Array<{
    text: string;
    start: number;
    end: number;
    isAcousticSplit: boolean;
  }> = [];
  let prevCharIdx = 0;
  let prevTime = cStart;

  for (const sp of splitPoints) {
    if (sp.charIdx - prevCharIdx < 3 || L - sp.charIdx < 3) continue;
    const subStr = chars
      .slice(prevCharIdx, sp.charIdx)
      .join('')
      .replace(/^[，、\s]+/, '')
      .trim();
    if (subStr) {
      result.push({
        text: subStr,
        start: prevTime,
        end: sp.timeSec,
        isAcousticSplit: prevCharIdx > 0,
      });
    }
    prevCharIdx = sp.charIdx;
    prevTime = sp.timeSec;
  }

  const tailStr = chars
    .slice(prevCharIdx)
    .join('')
    .replace(/^[，、\s]+/, '')
    .trim();
  if (tailStr) {
    result.push({
      text: tailStr,
      start: prevTime,
      end: cEnd,
      isAcousticSplit: prevCharIdx > 0,
    });
  }

  return result.length > 0
    ? result
    : [{ text: cText, start: cStart, end: cEnd, isAcousticSplit: false }];
}

function buildWordSpans(
  cleanText: string,
  segId: string,
  startSec: number,
  endSec: number,
  baseConfidence: number,
  acousticThreshold = 0.72
): WordSpan[] {
  const tokens = segmentNaturalLanguageTokens(cleanText);
  const dur = Math.max(0.2, endSec - startSec);
  const step = dur / Math.max(1, tokens.length);

  return tokens.map((tok, idx) => {
    const isPunct = /^[^\u4e00-\u9fa5\u3040-\u30ffa-zA-Z0-9]+$/.test(tok);
    const conf = isPunct
      ? 0.99
      : Number(Math.min(0.98, Math.max(0.45, baseConfidence)).toFixed(2));
    return {
      id: `${segId}_w_${idx}`,
      text: tok,
      start: Number((startSec + idx * step).toFixed(2)),
      end: Number((startSec + (idx + 1) * step).toFixed(2)),
      confidence: conf,
      status: conf < acousticThreshold ? 'acoustic_uncertain' : 'normal',
    };
  });
}

async function runWhisperWithWordTimestampsAndFallback(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  transcriber: any,
  audioInput: Float32Array,
  resolvedWhisperLang: string,
  chunkLengthSec = 30,
  strideLengthSec = 5,
  asrBeamSize = 1
): Promise<{
  chunks: Array<{ text: string; timestamp?: [number, number | null] }>;
  isWordLevel: boolean;
}> {
  let output: {
    text?: string;
    chunks?: Array<{ text: string; timestamp?: [number, number | null] }>;
  } | null = null;
  let isWordLevel = false;

  const safeBeams = isIntegratedOrLowMemDevice
    ? 1
    : Math.max(1, Math.min(3, Math.round(asrBeamSize || 1)));
  const effectiveChunkSec =
    safeBeams === 1 ? Math.min(25, chunkLengthSec) : chunkLengthSec;
  const effectiveStrideSec =
    safeBeams === 1
      ? Math.min(3, strideLengthSec)
      : safeBeams >= 3
      ? Math.max(6, strideLengthSec)
      : strideLengthSec;

  try {
    output = await transcriber(audioInput, {
      language: resolvedWhisperLang,
      task: 'transcribe',
      return_timestamps: true,
      chunk_length_s: effectiveChunkSec,
      stride_length_s: effectiveStrideSec,
      num_beams: safeBeams,
    });
    isWordLevel = false;
  } catch {
    output = null;
  }

  if (!output || !Array.isArray(output.chunks) || output.chunks.length === 0) {
    try {
      output = await transcriber(audioInput, {
        language: resolvedWhisperLang,
        task: 'transcribe',
        return_timestamps: 'word',
        chunk_length_s: effectiveChunkSec,
        stride_length_s: effectiveStrideSec,
        num_beams: 1,
      });
      if (output && Array.isArray(output.chunks) && output.chunks.length > 0) {
        isWordLevel = true;
      }
    } catch {
      // ignore
    }
  }

  const fullText =
    typeof output?.text === 'string'
      ? output.text
      : (output?.chunks || []).map((c) => c.text).join('');
  const compRatio = estimateRepetitionCompressionRatio(fullText);

  if (compRatio > 2.4) {
    console.warn(
      `⚠️ [LocalASR] 偵測到壓縮比異常 (${compRatio} > 2.4)，啟動防跳針退火重試...`
    );
    releaseResidualGpuTensors(output);
    try {
      const retryOut = await transcriber(audioInput, {
        language: resolvedWhisperLang,
        task: 'transcribe',
        return_timestamps: true,
        chunk_length_s: Math.min(20, chunkLengthSec),
        stride_length_s: 3,
        temperature: 0.2,
      });
      if (
        retryOut &&
        Array.isArray(retryOut.chunks) &&
        retryOut.chunks.length > 0
      ) {
        output = retryOut;
        isWordLevel = false;
      }
    } catch {
      // keep first output
    }
  }

  const rawChunks: Array<{
    text: string;
    timestamp?: [number, number | null];
  }> =
    output && Array.isArray(output.chunks) && output.chunks.length > 0
      ? output.chunks.map((c) => ({
          text: c.text || '',
          timestamp: c.timestamp
            ? [c.timestamp[0], c.timestamp[1]]
            : undefined,
        }))
      : [
          {
            text: typeof output?.text === 'string' ? output.text : '',
            timestamp: [0, audioInput.length / 16000],
          },
        ];

  releaseResidualGpuTensors(output);

  return {
    chunks: rawChunks,
    isWordLevel,
  };
}

function normalizeWhisperChunksToWordItems(
  rawChunks: Array<{ text: string; timestamp?: [number, number | null] }>,
  windowStartSec: number,
  windowEndSec: number,
  fineTurns: FineDiarizationTurn[],
  isWordLevel: boolean
): RawWordTimestampItem[] {
  const windowDur = Math.max(0.4, windowEndSec - windowStartSec);
  const windowTurns = fineTurns.filter(
    (ft) => ft.end > windowStartSec && ft.start < windowEndSec
  );
  const speechMinStart =
    windowTurns.length > 0 ? windowTurns[0].start : windowStartSec;
  const speechMaxEnd =
    windowTurns.length > 0
      ? windowTurns[windowTurns.length - 1].end
      : windowEndSec;

  const switchTimes: number[] = [];
  for (let i = 1; i < windowTurns.length; i++) {
    if (windowTurns[i].speakerId !== windowTurns[i - 1].speakerId) {
      switchTimes.push(windowTurns[i].start);
    }
  }

  const rawItems: RawWordTimestampItem[] = [];

  if (isWordLevel) {
    for (let i = 0; i < rawChunks.length; i++) {
      const ch = rawChunks[i];
      const rawTok = toTw((ch.text || '').trim());
      if (!rawTok) continue;
      if (/^(字幕|請不吝點讚|訂閱轉發|謝謝觀看|Amara\.org)/i.test(rawTok)) {
        continue;
      }

      const relStart =
        typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
      const relEnd =
        typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
          ? ch.timestamp[1]
          : Math.min(windowDur, relStart + 0.24);

      const gStart = Number(
        Math.max(windowStartSec, windowStartSec + relStart).toFixed(3)
      );
      const gEnd = Number(
        Math.max(
          gStart + 0.04,
          Math.min(windowEndSec + 0.2, windowStartSec + relEnd)
        ).toFixed(3)
      );

      const subTokens = segmentNaturalLanguageTokens(rawTok);

      const subSpan = Math.max(0.04, gEnd - gStart);
      const step = subSpan / Math.max(1, subTokens.length);

      for (let k = 0; k < subTokens.length; k++) {
        const st = subTokens[k].trim();
        if (!st) continue;
        if (/^[，。？！、；：,.?!]+$/.test(st) && rawItems.length > 0) {
          const prev = rawItems[rawItems.length - 1];
          if (!/[，。？！、；：,.?!]$/.test(prev.text)) {
            prev.text += st[0];
          }
          continue;
        }
        rawItems.push({
          text: st,
          start: Number((gStart + k * step).toFixed(3)),
          end: Number((gStart + (k + 1) * step).toFixed(3)),
          confidence: 0.9,
        });
      }
    }
    return rawItems;
  }

  for (let i = 0; i < rawChunks.length; i++) {
    const ch = rawChunks[i];
    const nextCh = i + 1 < rawChunks.length ? rawChunks[i + 1] : null;
    const chEndRel =
      typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
        ? ch.timestamp[1]
        : typeof ch.timestamp?.[0] === 'number'
        ? ch.timestamp[0] + 2.5
        : 0;
    const nextStartRel =
      nextCh && typeof nextCh.timestamp?.[0] === 'number'
        ? nextCh.timestamp[0]
        : chEndRel + 99;
    const pauseToNextSec = Math.max(0, nextStartRel - chEndRel);
    const absChEnd = windowStartSec + chEndRel;
    const absNextStart = windowStartSec + nextStartRel;
    const hasSwitchBetween = switchTimes.some(
      (st) => st >= absChEnd - 0.25 && st <= absNextStart + 0.25
    );
    const isMidTurnContinuation =
      Boolean(nextCh) && !hasSwitchBetween && pauseToNextSec <= 1.25;

    const cleanClause = restoreMultilingualPunctuation(ch.text || '', {
      isMidTurnContinuation,
      nextChunkRawText: nextCh?.text || '',
      pauseToNextSec,
    });
    if (!cleanClause) continue;

    const relStart =
      typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
    const relEnd =
      typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
        ? ch.timestamp[1]
        : Math.min(windowDur, relStart + 3.6);

    let gStart = Number((windowStartSec + relStart).toFixed(3));
    let gEnd = Number(
      Math.max(gStart + 0.25, windowStartSec + relEnd).toFixed(3)
    );

    if (i === 0 && Math.abs(gStart - speechMinStart) <= 0.6) {
      gStart = Number(Math.max(windowStartSec, speechMinStart).toFixed(3));
    }
    if (
      i === rawChunks.length - 1 &&
      gEnd > speechMaxEnd &&
      gEnd - speechMaxEnd <= 1.2
    ) {
      gEnd = Number(Math.max(gStart + 0.25, speechMaxEnd).toFixed(3));
    }

    for (const st of switchTimes) {
      if (
        st > gStart + 0.45 &&
        gEnd > st &&
        gEnd - st <= 0.65 &&
        (st - gStart) / Math.max(0.2, gEnd - gStart) >= 0.68
      ) {
        gEnd = Number(st.toFixed(3));
        break;
      }
    }

    const clauses = splitTextIntoDiscourseClauses(cleanClause);
    const numClauses = clauses.length;
    if (numClauses === 0) continue;

    const internalSwitches = switchTimes.filter(
      (t) => t >= gStart + 0.20 && t <= gEnd - 0.20
    );
    const claimedSwitches = new Set<number>();

    const clauseBounds = new Float64Array(numClauses + 1);
    clauseBounds[0] = gStart;
    clauseBounds[numClauses] = gEnd;

    if (numClauses > 1 && internalSwitches.length === numClauses - 1) {
      for (let k = 0; k < internalSwitches.length; k++) {
        clauseBounds[k + 1] = internalSwitches[k];
        claimedSwitches.add(internalSwitches[k]);
      }
    } else if (numClauses > 1) {
      const totalChars = Math.max(
        1,
        clauses.reduce((acc, c) => acc + c.length, 0)
      );
      let charAcc = 0;
      for (let k = 0; k < numClauses - 1; k++) {
        charAcc += clauses[k].length;
        const rawSplit = gStart + (charAcc / totalChars) * (gEnd - gStart);
        const minAllowed = clauseBounds[k] + 0.18;
        const maxAllowed = gEnd - (numClauses - 1 - k) * 0.18;

        let snapped = rawSplit;
        let minDist = 0.85;
        let bestSw: number | null = null;
        for (const st of internalSwitches) {
          if (!claimedSwitches.has(st) && st >= minAllowed && st <= maxAllowed) {
            const d = Math.abs(st - rawSplit);
            if (d < minDist) {
              minDist = d;
              snapped = st;
              bestSw = st;
            }
          }
        }
        if (bestSw !== null) claimedSwitches.add(bestSw);
        clauseBounds[k + 1] = Math.max(
          minAllowed,
          Math.min(maxAllowed, snapped)
        );
      }
    }

    for (let cIdx = 0; cIdx < numClauses; cIdx++) {
      const cText = clauses[cIdx];
      const cStart = clauseBounds[cIdx];
      const cEnd = Math.max(cStart + 0.16, clauseBounds[cIdx + 1]);

      const midClauseSwitches = internalSwitches.filter(
        (st) =>
          !claimedSwitches.has(st) && st >= cStart + 0.28 && st <= cEnd - 0.28
      );

      const acousticSubClauses = splitClauseAtAcousticSwitches(
        cText,
        cStart,
        cEnd,
        midClauseSwitches
      );

      for (let sIdx = 0; sIdx < acousticSubClauses.length; sIdx++) {
        const sub = acousticSubClauses[sIdx];
        const tokens = tokenizeClausePreventingTailOrphan(sub.text);

        const spanDur = Math.max(0.14, sub.end - sub.start);
        const step = spanDur / Math.max(1, tokens.length);

        for (let k = 0; k < tokens.length; k++) {
          const tok = tokens[k].trim();
          if (!tok) continue;
          if (/^[，。？！、；：,.?!]+$/.test(tok) && rawItems.length > 0) {
            const prev = rawItems[rawItems.length - 1];
            if (!/[，。？！、；：,.?!]$/.test(prev.text)) {
              prev.text += tok[0];
            }
            continue;
          }
          rawItems.push({
            text: tok,
            start: Number((sub.start + k * step).toFixed(3)),
            end: Number((sub.start + (k + 1) * step).toFixed(3)),
            confidence: 0.84,
            isAcousticTurnBoundary:
              k === 0 && (cIdx > 0 || sub.isAcousticSplit),
          });
        }
      }
    }
  }

  return rawItems;
}

function unwarpVirtualTrackChunksToSegments(
  rawChunks: Array<{ text: string; timestamp?: [number, number | null] }>,
  timeMap: VirtualTrackMapItem[],
  speakerId: string,
  chunkIndex: number,
  batchIndex: number,
  acousticThreshold: number
): Segment[] {
  if (timeMap.length === 0 || rawChunks.length === 0) return [];

  const totalVirtualSec = timeMap[timeMap.length - 1].virtualEndSec;
  const unwarpedUnits: Array<{
    text: string;
    realStart: number;
    realEnd: number;
    speakerId: string;
    runnerUpSpeakerId: string;
    confidence: number;
    acousticMargin: number;
  }> = [];

  for (let i = 0; i < rawChunks.length; i++) {
    const ch = rawChunks[i];
    const cleanText = restoreMultilingualPunctuation(ch.text || '');
    if (!cleanText) continue;

    const vStart =
      typeof ch.timestamp?.[0] === 'number' ? ch.timestamp[0] : 0;
    const vEnd =
      typeof ch.timestamp?.[1] === 'number' && ch.timestamp[1] !== null
        ? ch.timestamp[1]
        : Math.min(totalVirtualSec, vStart + 3.5);

    const overlappingItems = timeMap.filter(
      (item) =>
        Math.min(vEnd, item.virtualEndSec) -
          Math.max(vStart, item.virtualStartSec) >
        0.08
    );

    if (overlappingItems.length === 0) {
      const vCenter = 0.5 * (vStart + vEnd);
      let nearest = timeMap[0];
      let minDist = Infinity;
      for (const item of timeMap) {
        const itemCenter = 0.5 * (item.virtualStartSec + item.virtualEndSec);
        const d = Math.abs(vCenter - itemCenter);
        if (d < minDist) {
          minDist = d;
          nearest = item;
        }
      }
      overlappingItems.push(nearest);
    }

    if (overlappingItems.length === 1) {
      const item = overlappingItems[0];
      const vSpan = Math.max(0.15, item.virtualEndSec - item.virtualStartSec);
      const rSpan = Math.max(0.15, item.realEndSec - item.realStartSec);

      const clampedVStart = Math.max(
        item.virtualStartSec,
        Math.min(item.virtualEndSec - 0.06, vStart)
      );
      const clampedVEnd = Math.max(
        clampedVStart + 0.08,
        Math.min(item.virtualEndSec, vEnd)
      );

      const relRatioStart = (clampedVStart - item.virtualStartSec) / vSpan;
      const relRatioEnd = (clampedVEnd - item.virtualStartSec) / vSpan;

      const realStart = Number(
        (item.realStartSec + relRatioStart * rSpan).toFixed(2)
      );
      const realEnd = Number(
        Math.max(
          realStart + 0.12,
          item.realStartSec + relRatioEnd * rSpan
        ).toFixed(2)
      );

      unwarpedUnits.push({
        text: cleanText,
        realStart,
        realEnd,
        speakerId: item.speakerId || speakerId,
        runnerUpSpeakerId:
          item.runnerUpSpeakerId || item.speakerId || speakerId,
        confidence: item.speakerConfidence,
        acousticMargin: item.speakerMargin,
      });
    } else {
      const clauses = splitTextIntoDiscourseClauses(cleanText);

      if (clauses.length >= overlappingItems.length) {
        let cCursor = 0;
        for (let k = 0; k < overlappingItems.length; k++) {
          const item = overlappingItems[k];
          const takeCount =
            k === overlappingItems.length - 1
              ? clauses.length - cCursor
              : Math.max(
                  1,
                  Math.round(clauses.length / overlappingItems.length)
                );
          const subText = restoreMultilingualPunctuation(
            clauses.slice(cCursor, cCursor + takeCount).join('')
          );
          cCursor += takeCount;
          if (!subText) continue;

          unwarpedUnits.push({
            text: subText,
            realStart: item.realStartSec,
            realEnd: item.realEndSec,
            speakerId: item.speakerId || speakerId,
            runnerUpSpeakerId:
              item.runnerUpSpeakerId || item.speakerId || speakerId,
            confidence: item.speakerConfidence,
            acousticMargin: item.speakerMargin,
          });
        }
      } else {
        let bestItem = overlappingItems[0];
        let maxOv = -1;
        for (const item of overlappingItems) {
          const ov =
            Math.min(vEnd, item.virtualEndSec) -
            Math.max(vStart, item.virtualStartSec);
          if (ov > maxOv) {
            maxOv = ov;
            bestItem = item;
          }
        }
        unwarpedUnits.push({
          text: cleanText,
          realStart: bestItem.realStartSec,
          realEnd: bestItem.realEndSec,
          speakerId: bestItem.speakerId || speakerId,
          runnerUpSpeakerId:
            bestItem.runnerUpSpeakerId || bestItem.speakerId || speakerId,
          confidence: bestItem.speakerConfidence,
          acousticMargin: bestItem.speakerMargin,
        });
      }
    }
  }

  const mergedUnits: typeof unwarpedUnits = [];
  for (const u of unwarpedUnits) {
    const prev = mergedUnits[mergedUnits.length - 1];
    if (
      prev &&
      prev.speakerId === u.speakerId &&
      u.realStart >= prev.realEnd - 0.1 &&
      u.realStart - prev.realEnd <= 0.4 &&
      prev.text.length + u.text.length <= 160
    ) {
      const needsComma = !/[，。？！、；：,.?!]$/.test(prev.text);
      prev.text = (needsComma ? `${prev.text}，` : prev.text) + u.text;
      prev.realEnd = Math.max(prev.realEnd, u.realEnd);
      prev.acousticMargin = Math.max(prev.acousticMargin, u.acousticMargin);
    } else {
      mergedUnits.push({ ...u });
    }
  }

  return mergedUnits.map((m, idx) => {
    const segId = `seg_c${chunkIndex}_tb${batchIndex}_s${idx}`;
    const isLocked =
      m.acousticMargin >= 0.16 && m.realEnd - m.realStart >= 1.2;
    const words = buildWordSpans(
      m.text,
      segId,
      m.realStart,
      m.realEnd,
      m.confidence,
      acousticThreshold
    );
    const fullText = words.map((w) => w.text).join('');
    return {
      id: segId,
      chunkIndex,
      speakerId: m.speakerId,
      runnerUpSpeakerId: m.runnerUpSpeakerId,
      acousticMargin: m.acousticMargin,
      logOddsMargin: isLocked ? 2.1 : 1.1,
      overlapRatio: 0.05,
      slotTier: isLocked
        ? 'tier1_locked_anchor'
        : 'tier2_candidate_constrained',
      start: m.realStart,
      end: m.realEnd,
      words,
      text: fullText,
      rawText: fullText,
      isCorrected: false,
      relistenCount: 0,
    };
  });
}

self.onmessage = async (event: MessageEvent) => {
  const data = event.data;

  if (data.type === 'UNLOAD_MODEL' || data.type === 'DISPOSE') {
    if (asrInstance && typeof asrInstance.dispose === 'function') {
      try {
        await asrInstance.dispose();
      } catch {
        // ignore
      }
    }
    asrInstance = null;
    loadedModelKey = null;
    self.postMessage({ type: 'UNLOAD_DONE' });
    return;
  }

  if (data.type === 'TRANSCRIBE_SPEAKER_TRACK_BATCH') {
    const {
      stitchedSamples,
      timeMap,
      speakerId,
      chunkIndex,
      batchIndex,
      modelName,
      granularityTier = 'balanced_pro',
      languageHint = 'zh',
      acousticThreshold = 0.72,
    } = data as {
      stitchedSamples: Float32Array;
      timeMap: VirtualTrackMapItem[];
      speakerId: string;
      chunkIndex: number;
      batchIndex: number;
      modelName: LocalAsrModel;
      granularityTier?: GranularityTier;
      languageHint?: string;
      acousticThreshold?: number;
    };

    try {
      const transcriber = await ensureAsrPipeline(modelName, granularityTier);
      const audioInput =
        stitchedSamples instanceof Float32Array
          ? stitchedSamples
          : new Float32Array(stitchedSamples);

      if (audioInput.length < 1600 || !timeMap || timeMap.length === 0) {
        self.postMessage({
          type: 'SPEAKER_BATCH_TRANSCRIBED',
          segments: [],
          batchIndex,
        });
        return;
      }

      const resolvedWhisperLang =
        languageHint === 'en'
          ? 'english'
          : languageHint === 'ja'
          ? 'japanese'
          : 'chinese';

      const { chunks: rawChunks } =
        await runWhisperWithWordTimestampsAndFallback(
          transcriber,
          audioInput,
          resolvedWhisperLang,
          30,
          5
        );

      const unwarpedSegments = unwarpVirtualTrackChunksToSegments(
        rawChunks,
        timeMap,
        speakerId,
        chunkIndex,
        batchIndex,
        acousticThreshold
      );

      self.postMessage({
        type: 'SPEAKER_BATCH_TRANSCRIBED',
        segments: unwarpedSegments,
        batchIndex,
      });
    } catch (err) {
      self.postMessage({
        type: 'ASR_ERROR',
        error: (err as Error).message || '語者虛擬音軌轉寫發生錯誤',
      });
    }
    return;
  }

  if (data.type === 'TRANSCRIBE_WINDOW_AND_ALIGN') {
    const {
      windowSamples,
      windowStartSec,
      windowEndSec,
      fineTurns,
      posteriorMatrix,
      chunkIndex,
      windowIndex,
      modelName,
      granularityTier = 'balanced_pro',
      languageHint = 'zh',
      acousticThreshold = 0.72,
      asrBeamSize = 2,
      backchannelMarginThreshold = 0.12,
      pingPongMaxDurSec = 1.5,
    } = data as {
      windowSamples: Float32Array;
      windowStartSec: number;
      windowEndSec: number;
      fineTurns: FineDiarizationTurn[];
      posteriorMatrix?: FrameSpeakerPosteriorMatrix;
      chunkIndex: number;
      windowIndex: number;
      modelName: LocalAsrModel;
      granularityTier?: GranularityTier;
      languageHint?: string;
      acousticThreshold?: number;
      asrBeamSize?: number;
      backchannelMarginThreshold?: number;
      pingPongMaxDurSec?: number;
      engineMode?: string;
    };

    try {
      const transcriber = await ensureAsrPipeline(modelName, granularityTier);
      const audioInput =
        windowSamples instanceof Float32Array
          ? windowSamples
          : new Float32Array(windowSamples);

      if (audioInput.length < 1600) {
        self.postMessage({
          type: 'WINDOW_TRANSCRIBED',
          segments: [],
          windowIndex,
        });
        return;
      }

      const resolvedWhisperLang =
        languageHint === 'en'
          ? 'english'
          : languageHint === 'ja'
          ? 'japanese'
          : 'chinese';

      const { chunks: rawChunks, isWordLevel } =
        await runWhisperWithWordTimestampsAndFallback(
          transcriber,
          audioInput,
          resolvedWhisperLang,
          30,
          5,
          asrBeamSize
        );

      const wordItems = normalizeWhisperChunksToWordItems(
        rawChunks,
        windowStartSec,
        windowEndSec,
        fineTurns || [],
        isWordLevel
      );

      if (wordItems.length === 0) {
        self.postMessage({
          type: 'WINDOW_TRANSCRIBED',
          segments: [],
          windowIndex,
        });
        return;
      }

      const activePosteriorMatrix: FrameSpeakerPosteriorMatrix =
        fineTurns && fineTurns.length > 0
          ? synthesizePosteriorMatrixFromFineTurns(
              fineTurns,
              windowStartSec,
              windowEndSec,
              0.02
            )
          : posteriorMatrix &&
            posteriorMatrix.numFrames > 0 &&
            posteriorMatrix.speakerIds.length > 0
          ? posteriorMatrix
          : synthesizePosteriorMatrixFromFineTurns(
              [],
              windowStartSec,
              windowEndSec,
              0.02
            );

      const alignedWordSlots = decodeWordLevelSpeakersViterbi(
        wordItems,
        activePosteriorMatrix,
        {
          lambdaTrans: 0.55,
          backchannelMarginThreshold,
          pingPongMaxDurSec,
        }
      );

      const finalSegments = groupAlignedWordsIntoSegments(
        alignedWordSlots,
        chunkIndex,
        windowIndex,
        acousticThreshold
      );

      self.postMessage({
        type: 'WINDOW_TRANSCRIBED',
        segments: finalSegments,
        windowIndex,
      });
    } catch (err) {
      self.postMessage({
        type: 'ASR_ERROR',
        error: (err as Error).message || '語音轉寫發生錯誤',
      });
    }
    return;
  }

  if (data.type === 'RELISTEN_SUB_SEGMENT') {
    const {
      subSamples,
      targetSegment,
      modelName,
      granularityTier = 'balanced_pro',
      languageHint = 'zh',
    } = data as {
      subSamples: Float32Array;
      targetSegment: Segment;
      modelName: LocalAsrModel;
      granularityTier?: GranularityTier;
      languageHint?: string;
      promptHint?: string;
    };

    try {
      const transcriber = await ensureAsrPipeline(modelName, granularityTier);
      const audioInput =
        subSamples instanceof Float32Array
          ? subSamples
          : new Float32Array(subSamples);

      if (audioInput.length < 1600) {
        self.postMessage({
          type: 'RELISTEN_ERROR',
          segmentId: targetSegment.id,
          error: '音訊過短',
        });
        return;
      }

      const resolvedWhisperLang =
        languageHint === 'en'
          ? 'english'
          : languageHint === 'ja'
          ? 'japanese'
          : 'chinese';

      const { chunks } = await runWhisperWithWordTimestampsAndFallback(
        transcriber,
        audioInput,
        resolvedWhisperLang,
        20,
        3
      );

      const joinedRaw = chunks.map((c) => c.text).join('');
      const cleanText = restoreMultilingualPunctuation(joinedRaw);
      if (!cleanText) {
        self.postMessage({
          type: 'RELISTEN_ERROR',
          segmentId: targetSegment.id,
          error: '無文字產出',
        });
        return;
      }

      const newWords = buildWordSpans(
        cleanText,
        targetSegment.id,
        targetSegment.start,
        targetSegment.end,
        0.93
      );
      self.postMessage({
        type: 'RELISTEN_SUCCESS',
        segmentId: targetSegment.id,
        newWords,
      });
    } catch (err) {
      self.postMessage({
        type: 'RELISTEN_ERROR',
        segmentId: targetSegment.id,
        error: (err as Error).message,
      });
    }
  }
};
